/**
 * Created by nkssai on 10/8/14.
 */

import java.io.*;
import java.util.ArrayList;
import java.io.File;
import java.io.FileInputStream;

public class Model {
    /**
     * model parameters and inference
     */

    public int T;  // number of topics
    public int V;  // number of video
    public int W;  // vocabulary size

    // hyperparameters
    public float alpha;
    public float beta;
    public float betaB;
    public float gamma;

    // model parameters
    public int iterNum;  // number of Gibbs sampling iteration

    // real parameters
    public float[][] theta;  //video-topic distribution, V*T
    public float[] phi;  // y distribution, is 1 or 0, 1*2
    public float[] phiB;  // background word distribution, 1*V
    public float[][] vPhi;  // topic-word distribution, T * V

    public Model(int topicNum, float alphaP, float betaP, float betaBP, float gammaP) {
        this.T = topicNum;
        this.alpha = alphaP;
        this.beta = betaB;
        this.betaB = betaBP;
        this.gamma = gammaP;
    }
//
//    public boolean init(ArrayList<Integer> docs, int vSize) {
//        this.V = docs.size();
//        this.W = vSize;
//
//        Z = new int[V][];
//        for (int i = 0; i < V; i++)
//            for (i = )
//    }

    public void loadData(String path) {
        ArrayList<ArrayList> docs = new ArrayList<ArrayList>();
        File file = new File(path);
        String encoding = "utf-8";
        if (file.isFile() && file.exists()) { //判断文件是否存在
            InputStreamReader read = null;//考虑到编码格式
            try {
                read = new InputStreamReader(new FileInputStream(file), encoding);
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                while ((lineTxt = bufferedReader.readLine()) != null) {
                    System.out.println(lineTxt);
                }
                read.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void inference(){

    }
}
