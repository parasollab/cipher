
#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Multi-Robot SYCLOP Build Script${NC}"
echo -e "${BLUE}================================${NC}"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set up install prefix for dependencies
INSTALL_PREFIX="${SCRIPT_DIR}/install"
mkdir -p "$INSTALL_PREFIX"

echo -e "\n${GREEN}Step 1: Building eigenpy${NC}"
echo "This will be installed to: $INSTALL_PREFIX"

# Build and install eigenpy
cd eigenpy
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DPYTHON_EXECUTABLE=$(which python3)
make -j4
make install
cd "$SCRIPT_DIR"

echo -e "\n${GREEN}Step 2: Building pinocchio${NC}"
echo "Dependencies: eigenpy (already built)"

# Build and install pinocchio
cd pinocchio
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DBUILD_PYTHON_INTERFACE=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_BENCHMARK=OFF
make -j4
make install
cd "$SCRIPT_DIR"

echo -e "\n${GREEN}Step 3: Building crocoddyl${NC}"
echo "Dependencies: eigenpy, pinocchio (already built)"

# Build and install crocoddyl
cd crocoddyl
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
  -DBUILD_PYTHON_INTERFACE=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_BENCHMARK=OFF
make -j4
make install
cd "$SCRIPT_DIR"

echo -e "\n${GREEN}Step 4: Building main project${NC}"

# Now build the main project with all dependencies in CMAKE_PREFIX_PATH
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$INSTALL_PREFIX" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF
make -j4

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "\nExecutable location: ${SCRIPT_DIR}/build/main_mr_syclop"
echo -e "\nTo run, you may need to set:"
echo -e "  export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:\$LD_LIBRARY_PATH"
echo -e "  export PYTHONPATH=${INSTALL_PREFIX}/lib/python3.12/site-packages:\$PYTHONPATH"
