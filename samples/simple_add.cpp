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

#include <iostream>

using namespace cl::sycl;

class multiply;

int main() {
    const size_t array_size = 4;

    std::array<cl_int, array_size> A{{1, 2, 3, 4}}, B{{1, 2, 3, 4}}, C{{0, 0, 0, 0}};


    /* The scope we create here defines the lifetime of the buffer object, in SYCL
     * the lifetime of the buffer object dictates synchronization using RAII. */
    {
        /* We can also create a queue that uses the default selector in
         * the queue's default constructor. */
        queue myQueue;

        /* We define a buffer in order to maintain data across the host and one or
         * more devices. We construct this buffer with the address of the data
         * defined above and a range specifying a single element. */

        buffer<cl_int, 1> buffA(A.data(), range<1>(array_size));
        buffer<cl_int, 1> buffB(B.data(), range<1>(array_size));
        buffer<cl_int, 1> buffC(C.data(), range<1>(array_size));

        std::cout << myQueue.get_device().get_info<cl::sycl::info::device::max_compute_units>() << std::endl;

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

            std::cout << "Range size: " << range<1>(array_size).size() << std::endl;
            cgh.parallel_for<class multiply>(range<1>(array_size), kern);
        });

        /* queue::wait() will block until kernel execution finishes,
         * successfully or otherwise. */
        myQueue.wait();

    }

    for (int i = 0; i < array_size; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

}
