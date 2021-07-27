package com.seulawah.tracking.tracker;

import com.seulawah.tracking.kalman.DataPoint;
import com.seulawah.tracking.kalman.EKF;
import com.seulawah.tracking.kalman.KF;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.opencv.core.Point;

import java.util.Vector;

public class Track {

    public Vector<Point> trace;
    public int track_id;
    public int skipped_frames;
    public Point prediction;
    public DataPoint track_data;
    public EKF ekf;
    public KF kf;

    public Track(Point pt, int id) {
        trace = new Vector<>();
        track_id = id;
        skipped_frames = 0;
        prediction = pt;
        RealVector pointVec;
        pointVec = new ArrayRealVector(new double[]
                {pt.x,pt.y}
        );
        track_data = new DataPoint(pointVec);
        ekf = new EKF();
        kf = new KF();
    }
}
