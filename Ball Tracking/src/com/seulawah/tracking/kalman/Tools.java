package com.seulawah.tracking.kalman;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Tools {
    public RealVector convert_cartesian_to_polar(final RealVector v){
        final double THRESH = 0.0001;
        RealVector polar_vec;

        final double px = v.getEntry(0);
        final double py = v.getEntry(1);

        final double rho = Math.sqrt(px*px + py*py);
        final double phi = Math.atan2(py, px);

        polar_vec = new ArrayRealVector(new double[] {rho, phi});
        return polar_vec;
    }

    public RealMatrix calculate_Jacobian(final RealVector v){
        final double THRESH = 0.0001;
        RealMatrix H;

        H = new Array2DRowRealMatrix(new double[][] {
                {0,0,0,0},
                {0,0,0,0}
        });

        final double px = v.getEntry(0);
        final double py = v.getEntry(1);
        final double vx = v.getEntry(2);
        final double vy = v.getEntry(3);

        final double d_squared = px * px + py * py;
        final double d = Math.sqrt(d_squared);
        final double d_cubed = d_squared * d;

        if (d >= THRESH){
            final  double h11 = px / d;
            final double h12 = py / d;
            final  double h21 = -py / d_squared;
            final  double h22 = px / d_squared;

            H = new Array2DRowRealMatrix(new double[][] {
                    {h11,h12,0,0},
                    {h21,h22,0,0}
            });
        }
        return H;
    }
}
