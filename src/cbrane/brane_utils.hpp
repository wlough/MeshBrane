#ifndef brane_utils_hpp
#define brane_utils_hpp

// #pragma once
#include <cmath> // std::sqrt
#include <Eigen/Dense>
#include <thread>
#include <chrono>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>
#include <iterator>


class manual_timer
{
    std::chrono::high_resolution_clock::time_point t0;
    double timestamp{ 0.0 };
public:
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    void stop() { timestamp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() * 1000.0; }
    const double & get() { return timestamp; }
};

struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float w, x, y, z; };
struct double3 { double x, y, z; };
struct uint2 { uint32_t x, y; };
struct uint3 { uint32_t x, y, z; };
struct uint4 { uint32_t x, y, z, w; };

double dot(const double3& v1, const double3& v2)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

double norm(const double3& v)
{
    return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

double3 cross(const double3& v1, const double3& v2)
{
    return double3{
        v1.y*v2.z - v1.z*v2.y,
        v1.z*v2.x - v1.x*v2.z,
        v1.x*v2.y - v1.y*v2.x
    };
}

double triprod(const double3& a, const double3& b, const double3& c)
{
    return dot(a, cross(b, c));
}

struct HalfEdge;
struct Vertex;
struct Face;

struct Vertex
{
    Eigen::Vector3f xyz;
    double3 normal;
    float4 rgba;
    HalfEdge* halfedge;
    double mass;


};

struct Face
{
    HalfEdge* halfedge;
    double3 normal;
};

struct HalfEdge
{
    Vertex* vertex;
    HalfEdge* twin;
    HalfEdge* next;
    HalfEdge* prev;
    Face* face;
};

struct HalfEdgeMesh
{
    std::vector<Vertex*> vertices; // All vertices in the mesh
    std::vector<HalfEdge*> halfedges; // All half-edges in the mesh
    std::vector<Face*> faces; // All faces in the mesh
};


#endif 
