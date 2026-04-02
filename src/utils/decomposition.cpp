#include "decomposition.h"

#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/spaces/SE3StateSpace.h>

GridDecompositionImpl::GridDecompositionImpl(const int length, const int dim, const ompl::base::RealVectorBounds& bounds)
        : ompl::control::GridDecomposition(length, dim, bounds) {}

void GridDecompositionImpl::project(const ompl::base::State* s, std::vector<double>& coord) const
{
    if (dimension_ == 2)
    {
        coord.resize(2);
        coord[0] = s->as<ompl::base::SE2StateSpace::StateType>()->getX();
        coord[1] = s->as<ompl::base::SE2StateSpace::StateType>()->getY();
    }
    else if (dimension_ == 3)
    {
        coord.resize(3);
        coord[0] = s->as<ompl::base::SE3StateSpace::StateType>()->getX();
        coord[1] = s->as<ompl::base::SE3StateSpace::StateType>()->getY();
        coord[2] = s->as<ompl::base::SE3StateSpace::StateType>()->getZ();
    }
}

void GridDecompositionImpl::sampleFullState(const ompl::base::StateSamplerPtr& sampler, const std::vector<double>& coord, ompl::base::State* s) const
{
    sampler->sampleUniform(s);
    if (dimension_ == 2)
    {
        s->as<ompl::base::SE2StateSpace::StateType>()->setXY(coord[0], coord[1]);
    }
    else if (dimension_ == 3)
    {
        s->as<ompl::base::SE3StateSpace::StateType>()->setXYZ(coord[0], coord[1], coord[2]);
    }
}

void GridDecompositionImpl::getNeighbors(int rid, std::vector<int>& neighbors) const
{
    neighbors.clear();

    if (dimension_ == 2)
    {
        // 2D grid: length_ x length_
        int row = rid / length_;
        int col = rid % length_;

        // Cardinal directions only: up, down, left, right
        const int drow[] = {-1, 1, 0, 0};
        const int dcol[] = {0, 0, -1, 1};

        for (int i = 0; i < 4; ++i)
        {
            int nrow = row + drow[i];
            int ncol = col + dcol[i];

            if (nrow >= 0 && nrow < length_ && ncol >= 0 && ncol < length_)
            {
                neighbors.push_back(nrow * length_ + ncol);
            }
        }
    }
    else if (dimension_ == 3)
    {
        // 3D grid: length_ x length_ x length_
        int z = rid / (length_ * length_);
        int y = (rid / length_) % length_;
        int x = rid % length_;

        // Cardinal directions only: ±x, ±y, ±z
        const int dx[] = {-1, 1, 0, 0, 0, 0};
        const int dy[] = {0, 0, -1, 1, 0, 0};
        const int dz[] = {0, 0, 0, 0, -1, 1};

        for (int i = 0; i < 6; ++i)
        {
            int nx = x + dx[i];
            int ny = y + dy[i];
            int nz = z + dz[i];

            if (nx >= 0 && nx < length_ && ny >= 0 && ny < length_ && nz >= 0 && nz < length_)
            {
                neighbors.push_back(nz * length_ * length_ + ny * length_ + nx);
            }
        }
    }
}