/* 
 * File:   timer.h
 * Author: mianlu
 *
 * Created on May 29, 2015, 2:49 PM
 */

#ifndef TIMER_H
#define	TIMER_H

#include <sys/time.h>

class Timer {
protected:
    float _t; //ms
public:

    Timer() :
    _t(0.0f) {
    };

    virtual ~Timer() {
    };

    virtual void go() = 0;
    virtual void stop() = 0;

    void reset() {
        _t = 0;
    }

    float report() const {
        return _t;
    }
};

class CPUTimer : public Timer {
private:
    struct timeval _start, _end;
public:

    CPUTimer() :
    _start(), _end() {
    }

    ~CPUTimer() {
    }

    void go() {
        gettimeofday(&_start, NULL);
    }

    void stop() {
        gettimeofday(&_end, NULL);
        _t += ((_end.tv_sec - _start.tv_sec)*1000.0f + (_end.tv_usec - _start.tv_usec) / 1000.0f);
    }
};

#endif	/* TIMER_H */

