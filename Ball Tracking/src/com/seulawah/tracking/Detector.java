package com.seulawah.tracking;

import org.opencv.core.*;
import org.opencv.imgproc.Moments;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;


public class Detector {
    public Vector<Rect> rectArray = new Vector<>();
    public Vector<Point> detections = new Vector<>();

    public void detect(Mat im, Mat img) {

        //--Start filling hole--//
        //3.)
        Mat imgray3 = new Mat();
        List<MatOfPoint> contours3 = new ArrayList<>();
        Mat hierarchy3 = new Mat();
        Mat thres3 = new Mat();

        Imgproc.cvtColor(im, imgray3, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(imgray3, thres3, 127, 255, 0);
        Imgproc.findContours(thres3, contours3, hierarchy3, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        for (int idx = 0; idx < contours3.size(); idx++) {
            List<MatOfPoint> contourList = new ArrayList<>();
            contourList.add(contours3.get(idx));
            Imgproc.drawContours(im, contourList, -1, new Scalar(0, 255, 0), Imgproc.FILLED);
        }
        //--End of filling hole--//

        //--Start of detect by contour area--//
        for (int idx=0; idx < contours3.size(); idx++) {
            Mat contourM = contours3.get(idx);
            if (Imgproc.contourArea(contourM) > CONFIG.MIN_BLOB_AREA && Imgproc.contourArea(contourM) < CONFIG.MAX_BLOB_AREA) {
                List <MatOfPoint> contourList = new ArrayList<>();
                contourList.add(contours3.get(idx));
                Imgproc.drawContours(im, contourList, -1, new Scalar(0, 255, 0), Imgproc.FILLED);

                Rect boundRect = Imgproc.boundingRect(contourM);
                this.rectArray.add(boundRect);

                Imgproc.rectangle(im, boundRect.tl(), boundRect.br(),  new Scalar(0, 0, 255), 2);
                Imgproc.rectangle(img, boundRect.tl(), boundRect.br(),  new Scalar(0, 0, 255), 2);

                Moments moments = Imgproc.moments(contours3.get(idx));

                Point pt = new Point(moments.m10 / (moments.m00), moments.m01 / (moments.m00));
                this.detections.add(pt);

                Imgproc.circle(im, pt, 1, new Scalar(255, 0, 0), 3);
                Imgproc.circle(img, pt, 1, new Scalar(255, 0, 0), 3);
            }
        }
        //--End of detect by contour area--//
    }

    public Vector<Rect> getRectArray() {
        return this.rectArray;
    }

    public Vector<Point> getDetections() {
        return this.detections;
    }

}
