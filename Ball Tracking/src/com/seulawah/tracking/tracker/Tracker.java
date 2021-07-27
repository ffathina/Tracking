package com.seulawah.tracking.tracker;

import com.seulawah.tracking.assignment.AssignmentOptimal;
import com.seulawah.tracking.kalman.DataPoint;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.opencv.core.Point;
import java.util.Vector;

public class Tracker extends JTracker{

    int TractID = 0;
    Vector<Integer> assigned_tracks = new Vector<>();

    public Tracker(double _dist_thres, int _max_allowed_skipped_frames, int _max_trace_length) {
        tracks = new Vector<>();
        dist_thres = _dist_thres;
        max_allowed_skipped_frames = _max_allowed_skipped_frames;
        max_trace_length = _max_trace_length;
        track_removed = 0;
    }

    double euclideanDist(Point pt2, Point pt1) {
        Point diff = new Point(pt2.x - pt1.x, pt2.y - pt1.y);
        return Math.sqrt(Math.pow(diff.x, 2) + Math.pow(diff.y, 2));
    }

    public void updateTrack(Vector<Point> detections) {
        // Create tracks if no tracks vector found
        if (tracks.size() == 0) {
            for (int i = 0; i < detections.size(); i++) {
                Track tr = new Track(detections.get(i), TractID++);
                tracks.add(tr);
            }
        }

        //Calculate cost using sum of square distance between predicted vs detected centroids
        int N = tracks.size();
        int M = detections.size();
        double[][] Cost = new double[N][M];
        assigned_tracks.clear();

        for (int i = 0; i < tracks.size(); i++) {
            for (int j = 0; j < detections.size(); j++) {
                Cost[i][j] = euclideanDist(tracks.get(i).prediction, detections.get(j));
            }
        }

        AssignmentOptimal APS = new AssignmentOptimal();
        APS.Solve(Cost, assigned_tracks);
        Vector<Integer> not_assigned_tracks = new Vector<>();

        //If tracks are not detected for long time, remove them
        for (int i = 0; i < assigned_tracks.size(); i++) {
            if (assigned_tracks.get(i) != -1) {
                if (Cost[i][assigned_tracks.get(i)] > dist_thres) {
                    assigned_tracks.set(i, -1);
                    not_assigned_tracks.add(i);
                }
            } else {
                tracks.get(i).skipped_frames++;
                not_assigned_tracks.add(i);
            }
        }

        for (int i = 0; i < tracks.size(); i++) {
            if (tracks.get(i).skipped_frames > max_allowed_skipped_frames) {
                tracks.remove(i);
                assigned_tracks.remove(i);
                track_removed++;
                i--;
            }
        }

        //Look for not_assigned_detections
        Vector<Integer> not_assigned_detections = new Vector<>();
        for (int i = 0; i < detections.size(); i++) {
            if (!assigned_tracks.contains(i)) {
                not_assigned_detections.add(i);
            }
        }

        //Start new tracks for not_assigned_tracks
        if (not_assigned_detections.size() > 0) {
            for (int i = 0; i < not_assigned_detections.size(); i++) {
                Track tr = new Track(detections.get(not_assigned_detections.get(i)), TractID++);
                tracks.add(tr);
            }
        }

        updateKalman(detections);
        //updateExtendedKalman(detections);
    }

    public void updateKalman(Vector<Point> detections) {
        if (detections.size() == 0) {
            for(int i = 0; i < assigned_tracks.size(); i++) {
                assigned_tracks.set(i, -1);
            }
        }

        for (int i = 0; i < assigned_tracks.size(); i++) {
            if(!tracks.get(i).kf.getInitialized()) {
                tracks.get(i).kf.start(tracks.get(i).track_data);
                tracks.get(i).prediction = tracks.get(i).kf.getPrediction();

            } else {
                tracks.get(i).kf.predictStep();
                tracks.get(i).prediction = tracks.get(i).kf.getPrediction();

                if (assigned_tracks.get(i) != -1) {
                    tracks.get(i).skipped_frames = 0;
                    RealVector vec = new ArrayRealVector(new double[]
                            {detections.get(assigned_tracks.get(i)).x, detections.get(assigned_tracks.get(i)).y}
                    );
                    tracks.get(i).kf.updateStep(new DataPoint(vec));
                    tracks.get(i).prediction = tracks.get(i).kf.getPrediction();

                    /*System.out.println("predict: " + tracks.get(i).prediction + " (" + assigned_tracks.get(i) + ")");
                    System.out.println("groundt: " + vec);
                    System.out.println("-----------------------------------------");*/

                } else {
                    RealVector vec = new ArrayRealVector(new double[]
                            {tracks.get(i).kf.getPrediction().x, tracks.get(i).kf.getPrediction().y}
                    );
                    tracks.get(i).kf.updateStep(new DataPoint(vec));
                    tracks.get(i).prediction = tracks.get(i).kf.getPrediction();

                    /*System.out.println("predict: " + tracks.get(i).prediction);
                    System.out.println("groundt: Lost object");
                    System.out.println("-----------------------------------------");*/
                }
            }


            if (tracks.get(i).trace.size() > max_trace_length) {
                for (int j = 0; j < tracks.get(i).trace.size() - max_trace_length; j++) {
                    tracks.get(i).trace.remove(j);
                }
            }

            tracks.get(i).trace.add(tracks.get(i).prediction);
            //tracks.get(i).kf.setLastResult(tracks.get(i).prediction);
        }
    }

    public void updateExtendedKalman(Vector<Point> detections) {
        if (detections.size() == 0) {
            for(int i = 0; i < assigned_tracks.size(); i++) {
                assigned_tracks.set(i, -1);
            }
        }

        for (int i = 0; i < assigned_tracks.size(); i++) {
            if(!tracks.get(i).ekf.getInitialized()) {
                tracks.get(i).ekf.start(tracks.get(i).track_data);
                tracks.get(i).prediction = tracks.get(i).ekf.getPrediction();

            } else {
                tracks.get(i).ekf.predictStep();
                tracks.get(i).prediction = tracks.get(i).ekf.getPrediction();

                if (assigned_tracks.get(i) != -1) {
                    tracks.get(i).skipped_frames = 0;
                    RealVector vec = new ArrayRealVector(new double[]
                            {detections.get(assigned_tracks.get(i)).x, detections.get(assigned_tracks.get(i)).y}
                    );
                    tracks.get(i).ekf.updateStep(new DataPoint(vec));
                    tracks.get(i).prediction = tracks.get(i).ekf.getPrediction();

                    /*System.out.println("predict: " + tracks.get(i).prediction + " (" + assigned_tracks.get(i) + ")");
                    System.out.println("groundt: " + vec);
                    System.out.println("-----------------------------------------");*/

                } else {
                    RealVector vec = new ArrayRealVector(new double[]
                            {tracks.get(i).ekf.getPrediction().x, tracks.get(i).ekf.getPrediction().y}
                    );
                    tracks.get(i).ekf.updateStep(new DataPoint(vec));
                    tracks.get(i).prediction = tracks.get(i).ekf.getPrediction();

                    /*System.out.println("predict: " + tracks.get(i).prediction);
                    System.out.println("groundt: Lost object");
                    System.out.println("-----------------------------------------");*/
                }
            }


            if (tracks.get(i).trace.size() > max_trace_length) {
                for (int j = 0; j < tracks.get(i).trace.size() - max_trace_length; j++) {
                    tracks.get(i).trace.remove(j);
                }
            }

            tracks.get(i).trace.add(tracks.get(i).prediction);
            //tracks.get(i).ekf.setLastResult(tracks.get(i).prediction);
        }
    }
}
