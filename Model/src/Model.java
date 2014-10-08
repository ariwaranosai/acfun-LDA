/**
 * Created by nkssai on 10/8/14.
 */
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

    public void inference(){

    }
}
