/**
 * Created by nkssai on 10/8/14.
 */

import java.io.*;
import java.util.ArrayList;
import java.io.File;
import java.io.FileInputStream;
import java.util.List;
import java.util.Map;
import java.lang.*;
import java.lang.reflect.Array;

import net.sf.json.JSONArray;
import net.sf.json.JSONObject;
import net.sf.json.util.JSONUtils;

import common.MatrixUtil;
import sun.jvm.hotspot.jdi.ArrayReferenceImpl;

public class Model {
    /**
     * model parameters and inference
     */

    public int T;  // number of topics
    public int V;  // number of video
    public int W;  // vocabulary size

    // hyperparameters
    public double alpha;
    public double beta;
    public double betaB;
    public double gamma;

    // model parameters
    public int iterNum;  // number of Gibbs sampling iteration

    // real parameters
    public float[][] theta;  //video-topic distribution, V*T
    public float[] phi;  // y distribution, is 1 or 0, 1*2
    public float[] phiB;  // background word distribution, 1*V
    public float[][] vPhi;  // topic-word distribution, T * V


    // 采样参数
    public boolean y[][][]; //每一个词的生成指示
    public int Z[][]; //每一个帧的主题
    public int NW[]; // 背景词分布
    public int NTW[][]; // T x W, 每个主题的单词分布 sum is: SNTW[T]
    public long NY[]; // 1 x 2 y的分布
    public int NUT[][]; // sum U x T, 每个视频的主题分布 sum is: SNUT[U]
    public double SNTW[];
    public double SNUT[];

    public Model() {

    }

    public ArrayList<ArrayList<List>> document;

    public Model(int topicNum, float alphaP, double betaP, double betaBP, float gammaP) {
        this.T = topicNum;
        this.alpha = alphaP;
        this.beta = betaB;
        this.betaB = betaBP;
        this.gamma = gammaP;
    }

    /**
     * 推断过程
     * @param iteration
     * @param docs
     * @param saveStep 保持步长
     * @param saveTime 保存次数
     * @param buildIn build-in过程
     * @param outDir
     */
    public void inference(int iteration, ArrayList<ArrayList<List<Integer>>> docs, int saveStep,
                          int saveTime, int buildIn, String outDir) {
        if(iteration < saveStep * saveTime) {
            System.err.println("迭代次数过少");
            System.exit(0);
        }

        for (int i = 0 ; i < iteration; i++) {

            //TODO 加入一定迭代次数的校验

            for(int videoIndex = 0; videoIndex < V; videoIndex++) {
                //采样 p(z{{u,n}|c_-{u,n},w)
                for (int n = 0; n < docs.get(videoIndex).size(); n++) {
                    SampleTopic(docs.get(videoIndex).get(n), videoIndex, n);
                    for (int l = 0; l < docs.get(videoIndex).get(n).size(); l++)
                        SampleLabel(docs.get(videoIndex).get(n).get(l), videoIndex, n, l);
                }
            }

            if (i >= (iteration - 1 - (saveStep * (saveTime - 1))))
                if ((iteration - i - 1) % saveStep == 0) {
                    System.out.println("Saveing the model at " + (i + 1)
                            + "-th iteration");
                    try {
                        saveModelRes(outDir + "/model-" + (i + 1));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    no++;
                }
        }

    }

    private void SampleLabel(Integer word, int videoIndex, int n, int l) {
        if (y[videoIndex][n][l]) {
            NY[1]--;
            NTW[Z[videoIndex][n]][word]--;
            SNTW[Z[videoIndex][n]]--; // important!!
        } else {
            NY[0]--;
            NW[word]--;
        }

        double pt[] = new double[2];

        //计算 y = 1 y = 0
        double p0 = (double) (NY[0] + gamma) / (NY[0] + NY[1] + 2 * gamma);
        double p2 = p2 = (double) (NW[word] + betaB) / (NY[0] + V * betaB);

        double p3 = 1.0d;
        double sumRow = SNTW[Z[videoIndex][n]];

        p3 = (double) (NTW[Z[videoIndex][n]][word] + beta) / (sumRow + V * beta);

        pt[0] = p0 * p2;
        pt[1] = pt[0] + (1 - p0) * p3;

        // 模拟二项分布

        //考虑非归一化的情况
        double rouletter = (double) (Math.random() * pt[1]);
        int sample = 0;
        for (sample = 0; sample < 2; sample++) {
            if (pt[sample] > rouletter)
                break;
        }

        if (sample > 1) {
            System.err.println(" rand: \t" + rouletter);
            sample = (int) Math.round(Math.random());
        }

        //更新统计量
        if (sample == 1) {
            NY[1]++;
            NTW[Z[videoIndex][n]][word]++;
            SNTW[Z[videoIndex][n]]++;
            y[videoIndex][n][l] = true;
        } else {
            NY[0]++;
            NW[word]++;
            y[videoIndex][n][l] = false;
        }
    }

    private void SampleTopic(List<Integer> commit, int videoIndex, int screenShot) {
        int topic = Z[videoIndex][screenShot];
        //获取单词以及单词数
        ArrayList<Integer> tempWords = new ArrayList<Integer>();
        ArrayList<Integer> tempCounts = new ArrayList<Integer>();
        uniqe(commit, y[videoIndex][screenShot], tempWords, tempCounts);

        //更新 NTW每个主题的单词分布 NTU每个视频的主题
        if(NUT[videoIndex][topic] == 0) {
            System.err.println("error: NUT " + NUT[videoIndex][topic]);
            System.exit(0);
        }

        NUT[videoIndex][topic] --;
        SNUT[videoIndex] --;
        for (int j = 0; j < tempWords.size(); j++) {
            NTW[topic][tempWords.get(j)] -= tempCounts.get(j);
            SNTW[topic] -= tempCounts.get(j);
        }

        // 求p(Z_{u,n}|Z_c, W, Y, I)
        double[] pt = new  double[T];
        double NUTsumRowU = SNUT[videoIndex];

        for(int i = 0; i< T; i++) {
            int wcount = 0;
            double p1 = (double) (NUT[videoIndex][i] + alpha) / (NUTsumRowU + T * alpha);
            double p2 = 1.0D;
            for (int w = 0; w < tempWords.size(); w++) {
                int temvalue = NTW[i][tempWords.get(w)];
                double sumRow = SNTW[i];

                for (int numC = 0; numC < tempCounts.get(w); numC++) {
                    p2 = p2 * ((double) (temvalue + beta + numC) / ((double) sumRow + V * beta + wcount));
                    wcount++;
                }
            }
            pt[i] = p1 * p2;
        }

        // 计算多项分布的参数
        double rand = Math.random();
        double rouletter = (double) (rand * pt[T - 1]);
        int sample = 0;
        for (sample = 0; sample < T; sample++) {
            if (pt[sample] > rouletter)
                break;
        }
        if (sample > T - 1) {
            for (int i = 1; i < T; i++) {
                System.err.print(pt[i] + "\t");
            }
            System.err.println(" rand: \t" + rouletter);
            sample = (int) Math.round(Math.random() * (T - 1));
        }

        Z[videoIndex][topic] = sample;

        //更新其他统计量
        topic = sample;

        NUT[videoIndex][topic]++;
        SNUT[videoIndex]++;
        for (int w1 = 0; w1 < tempWords.size(); w1++) {
            NTW[topic][tempWords.get(w1)] += tempCounts.get(w1);
            SNTW[topic] += tempCounts.get(w1);
        }

        tempCounts.clear();
        tempWords.clear();
    }

    private void uniqe(List<Integer> words, boolean[] y, ArrayList<Integer> tempWords, ArrayList<Integer> tempCounts) {

        for (int i = 0; i < words.size(); i++) {
            if(y[i]) {
                if (tempWords.contains(i)) {
                    int index = tempWords.indexOf(words.get(i));
                    tempCounts.set(index, tempCounts.get(index) + 1);
                } else {
                    tempWords.add(words.get(i));
                    tempCounts.add(1);
                }
            }
        }
    }

    /**
     *
     * @param docs 数据
     * @param vSize 单词个数
     * @return
     */
    public boolean init(ArrayList<ArrayList<List<Integer>>> docs, int vSize) {
        System.out.println("-------- init -----------");
        this.V = docs.size();
        this.W = vSize;

        //初始化主题
        Z = new int[V][];
        for (int i = 0; i < V; i++) {
            Z[i] = new int[docs.get(i).size()];
            for (int j = 0; j < docs.get(i).size(); j++) {
                Z[i][j] = (int) Math.floor(Math.random() * T);
                if (Z[i][j] < 0)
                    Z[i][j] = 0;
                if (Z[i][j] > T - 1)
                    Z[i][j] = (int) (T - 1);
            }
        }

        y = new boolean[V][][];
        for (int i = 0; i< V; i++) {
            y[i] = new boolean[docs.get(i).size()][];
            for (int j = 0; j < docs.get(i).size(); j++) {
                y[i][j] = new boolean[docs.get(i).get(j).size()];
                for (int k = 0; k < docs.get(i).get(j).size(); k ++) {
                    if (Math.random() > 0.5)
                        y[i][j][k] = true;
                    else
                        y[i][j][k] = false;
                }
            }
        }

        cleanTempParam(docs);
        //更新 NW[V] NY[2] NUT[U][T]
        computeTempParam(docs, Z, y);
        computeSum(docs, T);
        return true;
    }

    private void computeSum(ArrayList<ArrayList<List<Integer>>> docs, int T) {
        SNUT = new double[docs.size()];

        for (int i = 0; i < docs.size(); i++) {
            SNUT[i] = MatrixUtil.sumRow(NUT, i);
        }

        SNTW = new double[T];

        for (int t = 0; t < T; t++) {
            SNTW[t] = MatrixUtil.sumRow(NTW, t);
        }
    }

    private void computeTempParam(ArrayList<ArrayList<List<Integer>>> docs, int[][] newZ, boolean[][][] newy) {

        for (int i = 0; i < V; i++) {
            for (int j = 0; j < newZ[i].length; j++) {
                NUT[i][newZ[i][j]]++;
            }

            for (int j = 0; j < newy[i].length; j++) {
                for (int k = 0; k < newy[i][j].length; k++){

                    if (newy[i][j][k]) {
                        NTW[Z[i][j]][docs.get(i).get(j).get(k)]++;
                        NY[1]++;
                    } else {
                        NW[docs.get(i).get(j).get(k)] ++;
                        NY[0]++;
                    }
                }
            }
        }

    }

    /**
     * init parameters in compute
     * @param docs
     */
    private void cleanTempParam(ArrayList<ArrayList<List<Integer>>> docs) {
        NW = new int[W];
        for (int i = 0; i < W; i++) {
            NW[i] = 0;
        }
        NTW = new int[T][];
        vPhi = new float[T][];
        for (int t = 0; t < T; t++) {
            NTW[t] = new int[W];
            vPhi[t] = new float[W];
            for (int i = 0; i < W; i++) {
                NTW[t][i] = 0;
                vPhi[t][i] = 0.0f;
            }
        }

        NY = new long[2];
        phi = new float[2];
        NY[0] = 0;
        NY[1] = 0;
        phi[0] = 0.0f;
        phi[1] = 0.0f;

        NUT = new int[V][];
        theta = new float[V][T];
        for (int i = 0; i < V; i++) {
            NUT[i] = new int[T];
            theta[i] = new float[T];
            for (int t = 0; t < T; t++) {
                NUT[i][t] = 0;
                theta[i][t] = 0.0f;
            }
        }
    }

    public ArrayList<ArrayList<List<Integer>>> loadData(String path) {
        /*
        从处理过的文本载入数据
        数据是 video screenshot w三层结构
         */

        ArrayList<ArrayList<List<Integer>>> doc = new ArrayList<ArrayList<List<Integer>>>();
        System.out.println("----load Data-----");
        File file = new File(path);
        String encoding = "utf-8";
        if (file.isFile() && file.exists()) { //判断文件是否存在
            InputStreamReader read = null;
            try {
                read = new InputStreamReader(new FileInputStream(file), encoding);
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                while ((lineTxt = bufferedReader.readLine()) != null) {
                    ArrayList<List<Integer>> videoCi = new ArrayList<List<Integer>>();
                    JSONObject videoObj = JSONObject.fromObject(lineTxt);
                    Map<String, List> commit = (Map<String, List>) videoObj.get("ci");

                    for(String i : commit.keySet()) {
                        JSONArray list = (JSONArray) commit.get(i);
                        List<Integer> ciList = JSONArray.toList(list);
                        videoCi.add(ciList);
                    }
                    doc.add(videoCi);
                }
                read.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("no file");
        }
        System.out.println("----- load complete ------");
        return doc;
    }

    public static void main(String args[]) throws IOException {
        String path = "/Users/shihang/code/tmp/tmp/id_json.txt";
        int wordSize = 1867;
        Model m = new Model(4, 10, 0.01, 0.01, 20);
        ArrayList<ArrayList<List<Integer>>> doc = m.loadData(path);
        m.init(doc, wordSize);
    }

}
