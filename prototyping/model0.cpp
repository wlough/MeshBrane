#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"
// using namespace tinyply;

#include "brane-utils.hpp"


void read_ply_file(const std::string & filepath, const bool preload_into_memory = true)
{
    std::cout << "........................................................................\n";
    std::cout << "Now Reading: " << filepath << std::endl;

    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try
    {
        // For most files < 1gb, pre-loading the entire file upfront and wrapping it into a 
        // stream is a net win for parsing speed, about 40% faster. 
        if (preload_into_memory)
        {
            byte_buffer = read_file_binary(filepath);
            file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
        }
        else
        {
            file_stream.reset(new std::ifstream(filepath, std::ios::binary));
        }

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
        std::shared_ptr<tinyply::PlyData> vertices, normals, colors, texcoords, faces, tripstrip;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties 
        // like vertex position are hard-coded: 
        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", "a" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
        // arbitrary ply files, it is best to leave this 0. 
        try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 3); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // Tristrips must always be read with a 0 list size hint (unless you know exactly how many elements
        // are specifically in the file, which is unlikely); 
        try { tripstrip = file.request_properties_from_element("tristrips", { "vertex_indices" }, 0); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        manual_timer read_timer;

        read_timer.start();
        file.read(*file_stream);
        read_timer.stop();

        const float parsing_time = static_cast<float>(read_timer.get()) / 1000.f;
        std::cout << "\tparsing " << size_mb << "mb in " << parsing_time << " seconds [" << (size_mb / parsing_time) << " MBps]" << std::endl;

        if (vertices)   std::cout << "\tRead " << vertices->count  << " total vertices "<< std::endl;
        if (normals)    std::cout << "\tRead " << normals->count   << " total vertex normals " << std::endl;
        if (colors)     std::cout << "\tRead " << colors->count << " total vertex colors " << std::endl;
        if (texcoords)  std::cout << "\tRead " << texcoords->count << " total vertex texcoords " << std::endl;
        if (faces)      std::cout << "\tRead " << faces->count     << " total faces (triangles) " << std::endl;
        if (tripstrip)  std::cout << "\tRead " << (tripstrip->buffer.size_bytes() / tinyply::PropertyTable[tripstrip->t].stride) << " total indices (tristrip) " << std::endl;

        // Example One: converting to your own application types
        {
            const size_t numVerticesBytes = vertices->buffer.size_bytes();
            std::vector<float3> verts(vertices->count);
            std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
        }

        // Example Two: converting to your own application type
        {
            std::vector<float3> verts_floats;
            std::vector<double3> verts_doubles;
            if (vertices->t == tinyply::Type::FLOAT32) { /* as floats ... */ }
            if (vertices->t == tinyply::Type::FLOAT64) { /* as doubles ... */ }
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // Circular write-read
    // write_ply_example("example_cube");
    // read_ply_file("example_cube-ascii.ply");
    // read_ply_file("example_cube-binary.ply", true);
    // read_ply_file("./data/ply_files/dumbbell.ply", true);
    read_ply_file("dumbbell_ultracoarse.ply");

    return EXIT_SUCCESS;
}
