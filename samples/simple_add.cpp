/***************************************************************************
 *
 *  Copyright (C) 2016 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  accessor.cpp
 *
 *  Description:
 *    Sample code that illustrates how to make data available on a device
 *    using accessors in SYCL.
 *
 **************************************************************************/

#include <CL/sycl.hpp>

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <chrono>
#include <functional>
#include <unistd.h>

#include "../core/model/Particle.h"

using namespace cl::sycl;

class multiply;


const size_t N = 10000000;

void vector_addition_regular(std::vector<cl_int> vector, std::vector<cl_int> vector1, std::vector<cl_int> vector2);

void create_vectors(std::vector<cl_int> &A, std::vector<cl_int> &B) {
    std::random_device rnd_device;
    // Specify the engine and distribution.
    std::mt19937 mersenne_engine{rnd_device()};  // Generates random integers
    std::uniform_int_distribution<int> dist{1, 100};

    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    std::generate(begin(A), end(A), gen);
    std::generate(begin(B), end(B), gen);
}

void vector_addition_sycl(std::vector<cl_int> &A, std::vector<cl_int> &B, std::vector<cl_int> C) {

    auto start = std::chrono::high_resolution_clock::now();

    /* The scope we create here defines the lifetime of the buffer object, in SYCL
     * the lifetime of the buffer object dictates synchronization using RAII. */
    {
        /* We can also create a queue that uses the default selector in
         * the queue's default constructor. */
        queue myQueue;

        /* We define a buffer in order to maintain data across the host and one or
         * more devices. We construct this buffer with the address of the data
         * defined above and a range specifying a single element. */

        buffer<cl_int, 1> buffA(A.data(), range<1>(N));
        buffer<cl_int, 1> buffB(B.data(), range<1>(N));
        buffer<cl_int, 1> buffC(C.data(), range<1>(N));


        myQueue.submit([&](handler &cgh) {
            /* We define accessors for requiring access to a buffer on the host or on
             * a device. Accessors are are like pointers to data we can use in
             * kernels to access the data. When constructing the accessor you must
             * specify the access target and mode. SYCL also provides the
             * get_access() as a buffer member function, which only requires an
             * access mode - in this case access::mode::read_write.
             * (make_access<>() has a second template argument which defaults
             * to access::mode::global) */
            auto accA = buffA.get_access<access::mode::read_write>(cgh);
            auto accB = buffB.get_access<access::mode::read_write>(cgh);
            auto accC = buffC.get_access<access::mode::read_write>(cgh);

            auto kern = [=](item<1> item) {
                size_t id = item.get_linear_id();
                accC[id] = accA[id] + accB[id];
            };

            cgh.parallel_for<class multiply>(range<1>(N), kern);
        });

        /* queue::wait() will block until kernel execution finishes,
         * successfully or otherwise. */
        myQueue.wait();

    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "sycl duration: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() * 0.000001 << "ms"
              << std::endl;

}


int main() {
//    float4 a = {1.0, 2.0, 3.0, 4.0};
//    Particle<float4> particle = Particle<float4>(a, a, a);
//
//    particle.move(0.1);
//    std::cout <<
//              particle.getPosition().x() << " " <<
//              particle.getPosition().y() << " " <<
//              particle.getPosition().z() << " " <<
//              particle.getPosition().w() << " " <<
//              std::endl;

    std::vector<cl_int> A(N);
    std::vector<cl_int> B(N);
    std::vector<cl_int> C(N);
    create_vectors(A, B);


    vector_addition_regular(A, B, C);
    usleep(2000000);
    vector_addition_sycl(A, B, C);
}

void vector_addition_regular(std::vector<cl_int> A, std::vector<cl_int> B, std::vector<cl_int> C) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        C[i] = (A[i] * A[i] + B[i] * B[i]) / 1.27 * (A[i] - A[i] / B[i] + B[i]) / (A[i] + A[i] - B[i] / B[i]);
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "regular duration: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() * 0.000001 << "ms"
              << std::endl;
}
