package com.seulawah.tracking.tracker;

import java.util.Vector;

import com.seulawah.tracking.kalman.DataPoint;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;


public abstract class JTracker {
    public double dist_thres;
    public int max_allowed_skipped_frames;
    public int max_trace_length;
    public Vector<Track> tracks;
    public int track_removed;
    public abstract void updateTrack(Vector<Point> detections);
}