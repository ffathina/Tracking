package com.seulawah.tracking.assignment;

import java.util.Vector;

public class AssignmentOptimal {
    public void Solve(double[][] DistMatrix, Vector<Integer> Assignment) {
        int N = DistMatrix.length; // number of columns (tracks)
        int M = DistMatrix[0].length; // number of rows (measurements)
        int dim =  Math.max(N, M);

        // Init
        int[] match;

        HungarianAlg b = new HungarianAlg(DistMatrix);
        match = b.execute();

        // form result
        Assignment.clear();
        for (int x = 0; x < N; x++) {
            Assignment.add(match[x]);
        }
    }
}
