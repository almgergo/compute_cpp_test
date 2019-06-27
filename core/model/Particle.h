//
// Created by almasy on 6/27/19.
//

#ifndef MYTEST_PARTICLE_H
#define MYTEST_PARTICLE_H

template <class T>
class Particle {
private:
    T position, velocity, acceleration;

public:
    Particle(
            T position,
            T velocity,
            T acceleration) : position(position), velocity(velocity), acceleration(acceleration) {}

    void move(double deltaT) {
        position += velocity * deltaT;
    }

    void accelerate(double deltaT) {
        velocity += acceleration * deltaT;
    }

    T getPosition() { return position; }
};


#endif //MYTEST_PARTICLE_H
