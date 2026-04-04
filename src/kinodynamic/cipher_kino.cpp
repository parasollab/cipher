#include <iostream>
#include <ompl/util/Console.h>

#include <dynoplan/optimization/ocp.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <queue>
#include <set>
#include <boost/heap/d_ary_heap.hpp>
#include <ompl/util/RandomNumbers.h>

#include "robotStatePropagator.hpp"
#include "fclStateValidityChecker.hpp"

#include "dynoplan/optimization/multirobot_optimization.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;
    OMPL_INFORM("OMPL is linked correctly.");
    return 0;
}
