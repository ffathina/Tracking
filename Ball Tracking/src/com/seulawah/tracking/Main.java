package com.seulawah.tracking;


import com.seulawah.tracking.tracker.Tracker;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.io.File;
import java.util.*;
import java.util.List;

public class Main {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static Tracker tracker;
    public static Detector detector;

    public static void main(String[] args) {
        tracker = new Tracker(CONFIG.dist_thres, CONFIG.max_allowed_skipped_frames, CONFIG.max_trace_length);

        File directoryPath = new File(CONFIG.path);

        //---Directory ball images---
        File[] filesList = directoryPath.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg"));
        List<String> path = new ArrayList<>();

        for (int i =0; i < filesList.length; i++) {
            String[] name = filesList[i].toString().split("e");
            String new_name = "D:/Java/Project/OPENCV/Ball Tracking/data/image"+ i +".jpg";
            path.add(new_name);
        }

        for (String file : path) {
            Mat im = Imgcodecs.imread(file);
            Imgproc.resize(im, im, new Size(700, 700));

            Mat img = im.clone();

            Vector<Rect> rectArray;

            Vector<Point> detections;

            detector = new Detector();
            detector.detect(im, img);

            detections = detector.getDetections();
            rectArray = detector.getRectArray();

            //---Kalman Filter Tracker---
            if (rectArray.size() > 0) {
                tracker.updateTrack(detections);
            } else if (rectArray.size() == 0)  {
                tracker.updateKalman(detections);
            }

            //---Extended Kalman Filter Tracker---
            /*if (rectArray.size() > 0) {
                tracker.updateTrack(detections);
            } else if (rectArray.size() == 0)  {
                tracker.updateExtendedKalman(detections);
            }*/

            for (int k = 0; k < tracker.tracks.size(); k++) {
                int traceNum = tracker.tracks.get(k).trace.size();
                if (traceNum > 1) {
                    for (int jt = 1; jt < traceNum; jt++) {
                        Imgproc.line(
                                img,
                                tracker.tracks.get(k).trace.get(jt - 1),
                                tracker.tracks.get(k).trace.get(jt),
                                CONFIG.Colors[tracker.tracks.get(k).track_id % 9],
                                2, 4, 0);
                    }
                }
            }

            /*HighGui.namedWindow("im", HighGui.WINDOW_AUTOSIZE);
            HighGui.imshow("im", im);*/

            HighGui.namedWindow("img", HighGui.WINDOW_AUTOSIZE);
            HighGui.imshow("img", img);

            HighGui.waitKey(CONFIG.waitkey);
        }

    }
}