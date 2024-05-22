/**
 * @file brane_utils.cpp
 * @brief Utility functions and structures for the MeshBrane library.
 */
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
#include "brane_utils.hpp"
#include <fstream> // std::ifstream


FaceVertexList load_face_vertex_list_from_ply(const std::string & filepath)
{
    FaceVertexList mesh;
    std::cout << "........................................................................\n";
    std::cout << "Now Reading: " << filepath << std::endl;

    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try
    {
        // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a 
        // stream is a net win for parsing speed, about 40% faster. 
        
        file_stream.reset(new std::ifstream(filepath, std::ios::binary));
        

        if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + filepath);

        file_stream->seekg(0, std::ios::end);
        const float size_mb = file_stream->tellg() * float(1e-6);
        file_stream->seekg(0, std::ios::beg);

        tinyply::PlyFile file;
        file.parse_header(*file_stream);

        std::cout << "\t[ply_header] Type: " << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
        for (const auto & c : file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
        for (const auto & c : file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

        for (const auto & e : file.get_elements())
        {
            std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
            for (const auto & p : e.properties)
            {
                std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
                if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
                std::cout << std::endl;
            }
        }

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers. 
        // See examples below on how to marry your own application-specific data structures with this one. 
        std::shared_ptr<tinyply::PlyData> vertices, faces;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties 
        // like vertex position are hard-coded: 
        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
        // arbitrary ply files, it is best to leave this 0. 
        try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        file.read(*file_stream);
    
        if (vertices)   std::cout << "\tRead " << vertices->count  << " total vertices "<< std::endl;
        if (faces)      std::cout << "\tRead " << faces->count     << " total faces (triangles) " << std::endl;

        
        // Converting vertices to your own application types
        {
            const size_t numVerticesBytes = vertices->buffer.size_bytes();
            mesh.vertices.resize(vertices->count);
            std::memcpy(mesh.vertices.data(), vertices->buffer.get(), numVerticesBytes);
        }

        // Converting faces to your own application type
        {
            const size_t numFacesBytes = faces->buffer.size_bytes();
            mesh.faces.resize(faces->count);
            std::memcpy(mesh.faces.data(), faces->buffer.get(), numFacesBytes);
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    }

     return mesh;
}

