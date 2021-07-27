package com.seulawah.tracking.kalman;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class DataPoint {
    private boolean initialized;
    private RealVector raw;

    public DataPoint() {
        this.initialized = false;
    }

    public DataPoint(final RealVector raw) {
        this.set(raw);
    }

    public void set(final RealVector raw) {
        this.raw = raw;
        this.initialized = true;
    }

    public final RealVector get() {
        return this.raw;
    }

    public final RealVector getState(){
        RealVector state = new ArrayRealVector(4);

        state.setEntry(0, this.raw.getEntry(0));
        state.setEntry(1, this.raw.getEntry(1));
        state.setEntry(2, 0);
        state.setEntry(3, 0);
        return state;
    }
}