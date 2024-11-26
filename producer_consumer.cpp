#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>

std::queue<int> buffer;                // Shared buffer (queue) between producer and consumer
const unsigned int maxBufferSize = 10; // Maximum size of the buffer
std::mutex mtx;                        // Mutex to synchronize access to the buffer
std::condition_variable cv;            // Condition variable for signaling

void producer() {
    int value = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(mtx); // Lock the mutex
        cv.wait(lock, [] { return buffer.size() < maxBufferSize; }); // Wait until buffer has space

        buffer.push(value++); // Produce an item and add it to the buffer
        std::cout << "Produced: " << value - 1 << std::endl;

        lock.unlock(); // Unlock the mutex
        cv.notify_all(); // Notify the consumer
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate production delay
    }
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx); // Lock the mutex
        cv.wait(lock, [] { return !buffer.empty(); }); // Wait until buffer has data

        int value = buffer.front(); // Consume an item from the buffer
        buffer.pop();
        std::cout << "Consumed: " << value << std::endl;

        lock.unlock(); // Unlock the mutex
        cv.notify_all(); // Notify the producer
        std::this_thread::sleep_for(std::chrono::milliseconds(150)); // Simulate consumption delay
    }
}

int main() {
    std::thread producerThread(producer);
    std::thread consumerThread(consumer);

    producerThread.join();
    consumerThread.join();

    return 0;
}

