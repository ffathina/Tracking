package com.seulawah.tracking.kalman;

import com.seulawah.tracking.CONFIG;

import org.opencv.core.Point;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.MatrixUtils;

/** This class is contain all matrices, updated matrices,
 * and methods to run the KF algorithm. To run this algorithm
 * use start(data), predictStep() then updateStep(data) or process(data) */

public class KF {

    private final int n = 4;
    private final double ax = CONFIG.accel_noise;
    private final double ay = CONFIG.accel_noise;
    private boolean initialized;
    private RealMatrix P;
    private RealMatrix F;
    private RealMatrix Q;
    private RealMatrix R_;
    private RealMatrix H_;
    private Point pt;
    private KalmanFilter KF = new KalmanFilter();

    public KF() {
        this.initialized = false;
        this.R_ = new Array2DRowRealMatrix();
        this.F = MatrixUtils.createRealIdentityMatrix(n);
        this.Q = MatrixUtils.createRealIdentityMatrix(n).scalarMultiply(0);

        this.R_ = new Array2DRowRealMatrix(new double[][] {
                {0.0225, 0.0},
                {0, 0.0225}
        });

        this.P = new Array2DRowRealMatrix(new double[][] {
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1000.0, 0.0},
                {0.0, 0.0, 0.0, 1000.0},
        });

        this.H_ = new Array2DRowRealMatrix(new double[][] {
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0}
        });
    }

    public void updateQ(final double dt){
        final double dt2 = dt * dt;
        final double dt3 = dt * dt2;
        final double dt4 = dt * dt3;

        final double r11 = dt4 * this.ax / 4;
        final double r13 = dt3 * this.ax / 2;
        final double r22 = dt4 * this.ay / 4;
        final double r24 = dt3 * this.ay / 2;
        final double r31 = dt3 * this.ax / 2;
        final double r33 = dt2 * this.ax;
        final double r42 = dt3 * this.ay / 2;
        final double r44 = dt2 * this.ay;

        this.Q = new Array2DRowRealMatrix(new double[][] {
                {r11, 0.0, r13, 0.0},
                {0.0, r22, 0.0, r24},
                {r31, 0.0, r33, 0.0},
                {0.0, r42, 0.0, r44},
        });

        this.KF.setQ(Q);
    }

    public void start(final DataPoint data) {
        RealVector x = data.getState();
        this.KF.start(this.n, x, this.P, this.F, this.Q);
        this.initialized = true;
    }

    public void predictStep() {
        final double dt  = CONFIG.dt;
        this.updateQ(dt);
        this.KF.updateF(dt);
        this.KF.predict();
    }

    public void updateStep(final DataPoint data) {
        final RealVector z = data.get();
        final RealVector x = this.KF.get();

        RealVector Hx;
        RealMatrix R;
        RealMatrix H;

        H = this.H_;
        Hx = this.H_.operate(x);
        R = this.R_;

        this.KF.update(z, H, Hx, R);
    }

    public void compute(final DataPoint data) {
        predictStep();
        updateStep(data);
    }

    /**
     * Construct KF algorithm.
     *
     * @param data
     *              input is measurement vector i.e point(x,y).
     */

    public void process(final DataPoint data) {
        if (this.initialized) {
            this.compute(data);
        } else {
            this.start(data);
        }
    }

    public final Point getPrediction() {
        pt = new Point(this.KF.get().getEntry(0), this.KF.get().getEntry(1));
        return pt;
    }

    public final boolean getInitialized() {
        return this.initialized;
    }

}