/**********************************************************************************************
� 2020. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and
to permit others to do so.
This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
1.  Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.
3.  Neither the name of the copyright holder nor the names of its contributors may be used
to endorse or promote products derived from this software without specific prior
written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********************************************************************************************/

#include "region_fill.h"
#include "matar.h"
#include "mesh.h"
#include "material.h"
#include "state.h"
#include "region.h"
#include "mesh_io.h"
#include "string_utils.h"

#include <stdio.h>
#include <fstream>


// -----------------------------------------------------------------------------
// The function to read a voxel vtk file from Dream3d and intialize the mesh
// ------------------------------------------------------------------------------
void user_voxel_init(DCArrayKokkos<size_t>& elem_values,
                     double& dx,
                     double& dy,
                     double& dz,
                     double& orig_x,
                     double& orig_y,
                     double& orig_z,
                     size_t& num_elems_i,
                     size_t& num_elems_j,
                     size_t& num_elems_k,
                     double scale_x,
                     double scale_y,
                     double scale_z,
                     std::string mesh_file)
{
    std::string MESH = mesh_file; // user specified

    std::ifstream in;  // FILE *in;
    in.open(MESH);

    // check to see of a mesh was supplied when running the code
    if (in)
    {
        printf("\nReading the 3D voxel mesh: ");
        std::cout << MESH << std::endl;
    }
    else
    {
        std::cout << "\n\n**********************************\n\n";
        std::cout << " ERROR:\n";
        std::cout << " Voxel vtk input does not exist \n";
        std::cout << "**********************************\n\n" << std::endl;
        std::exit(EXIT_FAILURE);
    } // end if

    size_t i;           // used for writing information to file

    size_t num_points_i;
    size_t num_points_j;
    size_t num_points_k;

    std::string token;

    bool found = false;

    // look for POINTS
    i = 0;
    while (found == false) {
        std::string str;
        std::string delimiter = " ";
        std::getline(in, str);
        std::vector<std::string> v = split(str, delimiter);

        // looking for the following text:
        //      POINTS %d float
        if (v[0] == "DIMENSIONS")
        {
            num_points_i = std::stoi(v[1]);
            num_points_j = std::stoi(v[2]);
            num_points_k = std::stoi(v[3]);
            printf("Num voxel nodes read in = %zu, %zu, %zu\n", num_points_i, num_points_j, num_points_k);

            found = true;
        } // end if

        if (i > 1000)
        {
            printf("ERROR: Failed to find POINTS \n");
            break;
        } // end if

        i++;
    } // end while

    found = false;

    CArray<double> pt_coords_x(num_points_i);
    CArray<double> pt_coords_y(num_points_j);
    CArray<double> pt_coords_z(num_points_k);

    while (found == false) {
        std::string str;
        std::string str0;
        std::string delimiter = " ";
        std::getline(in, str);
        std::vector<std::string> v = split(str, delimiter);

        // looking for the following text:
        if (v[0] == "X_COORDINATES")
        {
            size_t num_saved = 0;

            while (num_saved < num_points_i - 1) {
                // get next line
                std::getline(in, str0);

                // remove starting and trailing spaces
                str = trim(str0);
                std::vector<std::string> v_coords = split(str, delimiter);

                // loop over the contents of the vector v_coords
                for (size_t this_point = 0; this_point < v_coords.size(); this_point++)
                {
                    pt_coords_x(num_saved) = scale_x*std::stod(v_coords[this_point]);
                    num_saved++;
                } // end for
            } // end while

            found = true;
        } // end if

        if (i > 1000)
        {
            printf("ERROR: Failed to find X_COORDINATES \n");
            break;
        } // end if

        i++;
    } // end while
    found = false;

    while (found == false) {
        std::string str;
        std::string str0;
        std::string delimiter = " ";
        std::getline(in, str);
        std::vector<std::string> v = split(str, delimiter);

        // looking for the following text:
        if (v[0] == "Y_COORDINATES")
        {
            size_t num_saved = 0;

            while (num_saved < num_points_j - 1) {
                // get next line
                std::getline(in, str0);

                // remove starting and trailing spaces
                str = trim(str0);
                std::vector<std::string> v_coords = split(str, delimiter);

                // loop over the contents of the vector v_coords
                for (size_t this_point = 0; this_point < v_coords.size(); this_point++)
                {
                    pt_coords_y(num_saved) = scale_y*std::stod(v_coords[this_point]);
                    num_saved++;
                } // end for
            } // end while

            found = true;
        } // end if

        if (i > 1000)
        {
            printf("ERROR: Failed to find Y_COORDINATES \n");
            break;
        } // end if

        i++;
    } // end while
    found = false;

    while (found == false) {
        std::string str;
        std::string str0;
        std::string delimiter = " ";
        std::getline(in, str);
        std::vector<std::string> v = split(str, delimiter);

        // looking for the following text:
        if (v[0] == "Z_COORDINATES")
        {
            size_t num_saved = 0;

            while (num_saved < num_points_k - 1) {
                // get next line
                std::getline(in, str0);

                // remove starting and trailing spaces
                str = trim(str0);
                std::vector<std::string> v_coords = split(str, delimiter);

                // loop over the contents of the vector v_coords
                for (size_t this_point = 0; this_point < v_coords.size(); this_point++)
                {
                    pt_coords_z(num_saved) = scale_z*std::stod(v_coords[this_point]);
                    num_saved++;
                } // end for
            } // end while

            found = true;
        } // end if

        if (i > 1000)
        {
            printf("ERROR: Failed to find Z_COORDINATES \n");
            break;
        } // end if

        i++;
    } // end while
    found = false;

    size_t num_elems;
    num_elems_i = num_points_i - 1;
    num_elems_j = num_points_j - 1;
    num_elems_k = num_points_k - 1;

    // center to center distance between first and last elem along each edge
    double Lx = (pt_coords_x(num_points_i - 2) - pt_coords_x(0));
    double Ly = (pt_coords_y(num_points_j - 2) - pt_coords_y(0));
    double Lz = (pt_coords_z(num_points_k - 2) - pt_coords_z(0));

    // spacing between elems
    dx = Lx / ((double) num_elems_i);
    dy = Ly / ((double) num_elems_j);
    dz = Lz / ((double) num_elems_k);

    // element mesh origin
    orig_x = 0.5 * (pt_coords_x(0) + pt_coords_x(1)),
    orig_y = 0.5 * (pt_coords_y(0) + pt_coords_y(1)),
    orig_z = 0.5 * (pt_coords_z(0) + pt_coords_z(1)),

    // look for CELLS
    i = 0;
    while (found == false) {
        std::string str;
        std::getline(in, str);

        std::string              delimiter = " ";
        std::vector<std::string> v = split(str, delimiter);

        // looking for the following text:
        //      CELLS num_elems size
        if (v[0] == "CELL_DATA")
        {
            num_elems = std::stoi(v[1]);
            printf("Num voxel elements read in %zu\n", num_elems);

            found = true;
        } // end if

        if (i > 1000)
        {
            printf("ERROR: Failed to find CELL_DATA \n");
            break;
        } // end if

        i++;
    } // end while
    found = false;

    // allocate memory for element voxel values
    elem_values = DCArrayKokkos<size_t>(num_elems, "elem_values");

    // reading the cell data
    while (found == false) {
        std::string str;
        std::string str0;

        std::string delimiter = " ";
        std::getline(in, str);
        std::vector<std::string> v = split(str, delimiter);

        // looking for the following text:
        if (v[0] == "LOOKUP_TABLE")
        {
            size_t num_saved = 0;

            while (num_saved < num_elems - 1) {
                // get next line
                std::getline(in, str0);

                // remove starting and trailing spaces
                str = trim(str0);
                std::vector<std::string> v_values = split(str, delimiter);

                // loop over the contents of the vector v_coords
                for (size_t this_elem = 0; this_elem < v_values.size(); this_elem++)
                {
                    // save integers (0 or 1) to host side
                    elem_values.host(num_saved) = std::stoi(v_values[this_elem]);
                    num_saved++;
                } // end for

                // printf(" done with one row of data \n");
            } // end while

            found = true;
        } // end if

        if (i > 1000)
        {
            printf("ERROR: Failed to find LOOKUP_TABLE data \n");
            break;
        } // end if

        i++;
    } // end while
    found = false;

    printf("\n");

    in.close();
} // end routine

/////////////////////////////////////////////////////////////////////////////
///
/// \fn fill_geometric_region
///
/// \brief a function to calculate whether to fill this element based on the 
/// input instructions.  The output is
///  = 0 then no, do not fill this element
///  = 1 then yes, fill this element
///
/// \param mesh is the simulation mesh
/// \param node_coords is the nodal position array
/// \param voxel_elem_mat_id are the voxel values on a structured i,j,k mesh 
/// \param region_fills are the instructions to paint state on the mesh
/// \param mesh_coords is the geometric center of the element or a node coordinates
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
size_t fill_geometric_region(const Mesh_t& mesh,
                             const DCArrayKokkos<size_t>& voxel_elem_mat_id,
                             const DCArrayKokkos<int>& object_ids,
                             const CArrayKokkos<RegionFill_t>& region_fills,
                             const ViewCArrayKokkos <double>& mesh_coords,
                             const double voxel_dx, 
                             const double voxel_dy, 
                             const double voxel_dz,
                             const double orig_x, 
                             const double orig_y, 
                             const double orig_z,
                             const size_t voxel_num_i, 
                             const size_t voxel_num_j, 
                             const size_t voxel_num_k,
                             const size_t f_id,
                             const size_t elem_gid)
{

    // default is not to fill the element
    size_t fill_this = 0;


    // for shapes with an origin (e.g., sphere and circle), accounting for the origin
    double dist_x = mesh_coords(0) - region_fills(f_id).origin[0];
    double dist_y = mesh_coords(1) - region_fills(f_id).origin[1];
    double dist_z = mesh_coords(2) - region_fills(f_id).origin[2];

    // spherical radius 
    double radius = sqrt(dist_x * dist_x +
                         dist_y * dist_y +
                         dist_z * dist_z);

    // cylindrical radius
    double radius_cyl = sqrt(dist_x * dist_x +
                             dist_y * dist_y);


    // check to see if this element should be filled
    switch (region_fills(f_id).volume) {
        case region::global:
            {
                fill_this = 1;
                break;
            }
        case region::box:
            {

                double x_lower_bound = region_fills(f_id).x1;
                double x_upper_bound = region_fills(f_id).x2;

                double y_lower_bound = region_fills(f_id).y1;
                double y_upper_bound = region_fills(f_id).y2;

                double z_lower_bound = region_fills(f_id).z1;
                double z_upper_bound = region_fills(f_id).z2;


                if (mesh_coords(0) >= x_lower_bound && mesh_coords(0) <= x_upper_bound &&
                    mesh_coords(1) >= y_lower_bound && mesh_coords(1) <= y_upper_bound &&
                    mesh_coords(2) >= z_lower_bound && mesh_coords(2) <= z_upper_bound) {
                    fill_this = 1;
                }
                break;
            }
        case region::cylinder:
            {
                double z_lower_bound = region_fills(f_id).z1;
                double z_upper_bound = region_fills(f_id).z2;

                if (radius_cyl >= region_fills(f_id).radius1 && 
                    radius_cyl <= region_fills(f_id).radius2 &&
                    mesh_coords(2) >= z_lower_bound && mesh_coords(2) <= z_upper_bound) {
                    fill_this = 1;
                }
                break;
            }
        case region::sphere:
            {
                if (radius >= region_fills(f_id).radius1
                    && radius <= region_fills(f_id).radius2) {
                    fill_this = 1;
                }
                break;
            }

        case region::readVoxelFile:
            {

                fill_this = 0; // default is no, don't fill it

                // find the closest element in the voxel mesh to this element
                double i0_real = (mesh_coords(0) - orig_x - region_fills(f_id).origin[0]) / (voxel_dx);
                double j0_real = (mesh_coords(1) - orig_y - region_fills(f_id).origin[1]) / (voxel_dy);
                double k0_real = (mesh_coords(2) - orig_z - region_fills(f_id).origin[2]) / (voxel_dz);

                int i0 = (int)i0_real;
                int j0 = (int)j0_real;
                int k0 = (int)k0_real;

                // look for the closest element in the voxel mesh
                int elem_id0 = get_id_device(i0, j0, k0, voxel_num_i, voxel_num_j);

                // if voxel mesh overlaps this mesh, then fill it if =1
                if (elem_id0 < voxel_elem_mat_id.size() && elem_id0 >= 0 &&
                    i0 >= 0 && j0 >= 0 && k0 >= 0 &&
                    i0 < voxel_num_i && j0 < voxel_num_j && k0 < voxel_num_k) {
                    // voxel mesh elem values = 0 or 1
                    fill_this = voxel_elem_mat_id(elem_id0); // values from file

                } // end if

                break;

            } // end case
        case region::readVTUFile:
            {
                // if the part id in .vtu file matches the specified id, then fill it
                if(object_ids(elem_gid) == region_fills(f_id).part_id){
                    fill_this = 1;
                }
                break;
            }
        case region::no_volume:
            {
                fill_this = 0; // default is no, don't fill it

                break;
            }
        default:
            {
                fill_this = 0; // default is no, don't fill it

                break;
            }

    } // end of switch


    return fill_this;

} // end function



/////////////////////////////////////////////////////////////////////////////
///
/// \fn append_fills_in_elem
///
/// \brief a function to append fills 
///
/// \param elem_fill_ids is the fill id in an element
/// \param num_fills_saved_in_elem is the number of fills the element has
/// \param region_fills are the instructions to paint state on the mesh
/// \param elem_gid is the element global mesh index
/// \param fill_id is fill instruction
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void append_fills_in_elem(const DCArrayKokkos <double>& elem_volfracs,
                          const CArrayKokkos <size_t>& elem_fill_ids,
                          const DCArrayKokkos <size_t>& num_fills_saved_in_elem,
                          const CArrayKokkos<RegionFill_t>& region_fills,
                          const double combined_volfrac,
                          const size_t elem_gid,
                          const size_t fill_id)
{

    // the number of materials saved to this element, initialized to 0 at start of code
    size_t fill_storage_lid = num_fills_saved_in_elem(elem_gid);

    // check on exceeding 3 materials per element
    if (num_fills_saved_in_elem(elem_gid) > 3){
        Kokkos::abort("ERROR: exceeded 3 materials in an element when painting regions on the mesh \n");
    } // end if check


    // material id assigned to this fill
    const size_t mat_id = region_fills(fill_id).material_id;


    // check to see if the material already exists
    bool check_mat_exists = false;
    for (size_t a_fill=0; a_fill < num_fills_saved_in_elem(elem_gid); a_fill++){

        // get the mat_id in this fill
        size_t a_mat_id = region_fills(a_fill).material_id;
        if(mat_id == a_mat_id){
            // overwrite the existing material lid with new fill instructions
            fill_storage_lid = a_fill;  
            check_mat_exists = true;
        } // end if check on mat_id existing already

    } // end for a_fill


    // There will now be at least 1 material so we want
    // num_fills_saved_in_elem >= 1, and it is intialized at 0 
    if (check_mat_exists == false){
        // we are adding a new material, so increment the number of saved
        num_fills_saved_in_elem(elem_gid) += 1;
    } // end check if material is a new one


    // confirm the volume fractions in each elem tally to 1 later in the code

    // --- append the volfracs and fill ids in elem ---
    elem_fill_ids(elem_gid, fill_storage_lid) = fill_id;
    elem_volfracs(elem_gid, fill_storage_lid) = combined_volfrac;

    // done with calculating the fill instructions

} // end function painting fill ids


/////////////////////////////////////////////////////////////////////////////
///
/// \fn get_region_scalar
///
/// \brief a function to get the scalar field value
///
/// \param field_scalar is the field
/// \param mesh_coords are the coordinates of the elem/gauss/nodes
/// \param scalar value
/// \param slope value
/// \param mesh_gid is the elem/gauss/nodes global mesh index
/// \param num_dims is dimensions
/// \param scalar_field is an enum on how the field is to be calculated
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
double get_region_scalar(const ViewCArrayKokkos <double> mesh_coords,
                         const double scalar,
                         const double slope,
                         const size_t mesh_gid,
                         const size_t num_dims,
                         const init_conds::init_scalar_conds scalarFieldType)
{
    double value_out;

    // --- scalar field ---
    switch (scalarFieldType) {
        case init_conds::uniform:
            {
                value_out = scalar;
                break;
            }
        // radial in the (x,y) plane where x=r*cos(theta) and y=r*sin(theta)
        case init_conds::radialScalar:
            {
                // Setting up radial
                double dir[2];
                dir[0] = 0.0;
                dir[1] = 0.0;
                double radius_val = 0.0;

                for (int dim = 0; dim < 2; dim++) {
                    dir[dim]    = mesh_coords(dim);
                    radius_val += mesh_coords(dim) * mesh_coords(dim);
                } // end for
                radius_val = sqrt(radius_val);

                for (int dim = 0; dim < 2; dim++) {
                    if (radius_val > 1.0e-14) {
                        dir[dim] /= (radius_val);
                    }
                    else{
                        dir[dim] = 0.0;
                    }
                } // end for

                value_out = scalar * dir[0];
                value_out = scalar * dir[1];

                break;
            }
        case init_conds::sphericalScalar:
            {
                // Setting up spherical
                double dir[3];
                dir[0] = 0.0;
                dir[1] = 0.0;
                dir[2] = 0.0;
                double radius_val = 0.0;

                for (int dim = 0; dim < 3; dim++) {
                    dir[dim]    = mesh_coords(dim);
                    radius_val += mesh_coords(dim) * mesh_coords(dim);
                } // end for
                radius_val = sqrt(radius_val);

                for (int dim = 0; dim < 3; dim++) {
                    if (radius_val > 1.0e-14) {
                        dir[dim] /= (radius_val);
                    }
                    else{
                        dir[dim] = 0.0;
                    }
                } // end for

                value_out = scalar * radius_val;
                break;
            }
        case init_conds::xlinearScalar:
            {
                // scalar_field = slope*x + value
                value_out = slope*mesh_coords(0) + scalar;
                break;
            }
        case init_conds::ylinearScalar:
            {
                // scalar_field = slope*y + value
                value_out = slope*mesh_coords(1) + scalar;
                break;
            }
        case init_conds::zlinearScalar:
            {
                // scalar_field = slope*z + value
                value_out = slope*mesh_coords(2) + scalar;
                break;
            }
        case init_conds::tgVortexScalar:
            {
                printf("**** TG Vortex not supported for general scalar initial conditions ****\n");

                break;
            }
        case init_conds::noICsScalar:
            {
                // nothing is done

                break;
            }
        default:
            {
                // do nothing

                break;
            }
    } // end of switch

    return value_out;

}  // end function paint_scalar

/////////////////////////////////////////////////////////////////////////////
///
/// \fn paint_multi_scalar
///
/// \brief a function to paint multiple material scalars on the mesh
///
/// \param field_scalar is the field
/// \param mesh_coords are the coordinates of the elem/gauss/nodes
/// \param mesh_gid is the elem/gauss/nodes global mesh index
/// \param num_dims is dimensions
/// \param bin is for multiple materials at that location
/// \param scalar_field is an enum on how the field is to be set
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void paint_multi_scalar(const DCArrayKokkos<double>& field_scalar,
                        const ViewCArrayKokkos <double> mesh_coords,
                        const double scalar,
                        const double slope,
                        const size_t mesh_gid,
                        const size_t num_dims,
                        const size_t bin,
                        const init_conds::init_scalar_conds scalarFieldType)
{

    // --- scalar field ---
    switch (scalarFieldType) {
        case init_conds::uniform:
            {
                field_scalar(mesh_gid,bin) = scalar;
                break;
            }
        // radial in the (x,y) plane where x=r*cos(theta) and y=r*sin(theta)
        case init_conds::radialScalar:
            {
                // Setting up radial
                double dir[2];
                dir[0] = 0.0;
                dir[1] = 0.0;
                double radius_val = 0.0;

                for (int dim = 0; dim < 2; dim++) {
                    dir[dim]    = mesh_coords(dim);
                    radius_val += mesh_coords(dim) * mesh_coords(dim);
                } // end for
                radius_val = sqrt(radius_val);

                for (int dim = 0; dim < 2; dim++) {
                    if (radius_val > 1.0e-14) {
                        dir[dim] /= (radius_val);
                    }
                    else{
                        dir[dim] = 0.0;
                    }
                } // end for

                field_scalar(mesh_gid,bin) = scalar * dir[0];
                field_scalar(mesh_gid,bin) = scalar * dir[1];

                break;
            }
        case init_conds::sphericalScalar:
            {
                // Setting up spherical
                double dir[3];
                dir[0] = 0.0;
                dir[1] = 0.0;
                dir[2] = 0.0;
                double radius_val = 0.0;

                for (int dim = 0; dim < 3; dim++) {
                    dir[dim]    = mesh_coords(dim);
                    radius_val += mesh_coords(dim) * mesh_coords(dim);
                } // end for
                radius_val = sqrt(radius_val);

                for (int dim = 0; dim < 3; dim++) {
                    if (radius_val > 1.0e-14) {
                        dir[dim] /= (radius_val);
                    }
                    else{
                        dir[dim] = 0.0;
                    }
                } // end for

                field_scalar(mesh_gid,bin) = scalar * radius_val;
                break;
            }
        case init_conds::xlinearScalar:
            {
                // scalar_field = slope*x + value
                field_scalar(mesh_gid,bin) = slope*mesh_coords(0) + scalar;
                break;
            }
        case init_conds::ylinearScalar:
            {
                // scalar_field = slope*y + value
                field_scalar(mesh_gid,bin) = slope*mesh_coords(1) + scalar;
                break;
            }
        case init_conds::zlinearScalar:
            {
                // scalar_field = slope*z + value
                field_scalar(mesh_gid,bin) = slope*mesh_coords(2) + scalar;
                break;
            }
        case init_conds::tgVortexScalar:
            {
                printf("**** TG Vortex not supported for general scalar initial conditions ****\n");

                break;
            }
        case init_conds::noICsScalar:
            {
                // nothing is done

                break;
            }
        default:
            {
                // do nothing

                break;
            }
    } // end of switch

}  // end function paint_multi_scalar



/////////////////////////////////////////////////////////////////////////////
///
/// \fn paint_vector_rk
///
/// \brief a function to paint a vector fields on the mesh 
///
/// \param vector is the vector field on elem/gauss/node
/// \param coords are the coordinates of the mesh elem/guass/node
/// \param u is the x-comp
/// \param v is the y-comp
/// \param w is the z-comp
/// \param scalar is the magnitude
/// \param mesh_gid is the node global mesh index
/// \param rk_num_bins is time integration storage level
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void paint_vector_rk(const DCArrayKokkos<double>& vector_field,
                     const ViewCArrayKokkos <double>& mesh_coords,
                     const double u,
                     const double v,
                     const double w,
                     const double scalar,
                     const size_t mesh_gid,
                     const size_t num_dims,
                     const size_t rk_num_bins,
                     const init_conds::init_vector_conds vectorFieldType)
{

    // save vector field at all rk_levels
    for(size_t rk_level=0; rk_level<rk_num_bins; rk_level++){

        // --- vector ---
        switch (vectorFieldType) {
            case init_conds::cartesian:
                {
                    vector_field(rk_level, mesh_gid, 0) = u;
                    vector_field(rk_level, mesh_gid, 1) = v;
                    if (num_dims == 3) {
                        vector_field(rk_level, mesh_gid, 2) = w;
                    }

                    break;
                }
            // radial in the (x,y) plane where x=r*cos(theta) and y=r*sin(theta)
            case init_conds::radialVec:
                {
                    // Setting up radial
                    double dir[2];
                    dir[0] = 0.0;
                    dir[1] = 0.0;
                    double radius_val = 0.0;

                    for (int dim = 0; dim < 2; dim++) {
                        dir[dim]    = mesh_coords(dim);
                        radius_val += mesh_coords(dim) * mesh_coords(dim);
                    } // end for
                    radius_val = sqrt(radius_val);

                    for (int dim = 0; dim < 2; dim++) {
                        if (radius_val > 1.0e-14) {
                            dir[dim] /= (radius_val);
                        }
                        else{
                            dir[dim] = 0.0;
                        }
                    } // end for

                    vector_field(rk_level, mesh_gid, 0) = scalar * dir[0];
                    vector_field(rk_level, mesh_gid, 1) = scalar * dir[1];
                    if (num_dims == 3) {
                        vector_field(rk_level, mesh_gid, 2) = 0.0;
                    }

                    break;
                }
            case init_conds::sphericalVec:
                {
                    // Setting up spherical
                    double dir[3];
                    dir[0] = 0.0;
                    dir[1] = 0.0;
                    dir[2] = 0.0;
                    double radius_val = 0.0;

                    for (int dim = 0; dim < 3; dim++) {
                        dir[dim]    = mesh_coords(dim);
                        radius_val += mesh_coords(dim) * mesh_coords(dim);
                    } // end for
                    radius_val = sqrt(radius_val);

                    for (int dim = 0; dim < 3; dim++) {
                        if (radius_val > 1.0e-14) {
                            dir[dim] /= (radius_val);
                        }
                        else{
                            dir[dim] = 0.0;
                        }
                    } // end for

                    vector_field(rk_level, mesh_gid, 0) = scalar * dir[0];
                    vector_field(rk_level, mesh_gid, 1) = scalar * dir[1];
                    if (num_dims == 3) {
                        vector_field(rk_level, mesh_gid, 2) = scalar * dir[2];
                    }

                    break;
                }
            case init_conds::radialLinearVec:
                {
                    printf("**** Radial_linear initial conditions not yet supported ****\n");
                    break;
                }
            case init_conds::sphericalLinearVec:
                {
                    printf("**** spherical_linear initial conditions not yet supported ****\n");
                    break;
                }
            case init_conds::tgVortexVec:
                {
                    vector_field(rk_level, mesh_gid, 0) = sin(PI * mesh_coords(0)) * 
                                                        cos(PI * mesh_coords(1));
                    vector_field(rk_level, mesh_gid, 1) = -1.0 * cos(PI * mesh_coords(0)) * 
                                                        sin(PI * mesh_coords(1));
                    if (num_dims == 3) {
                        vector_field(rk_level, mesh_gid, 2) = 0.0;
                    }

                    break;
                }
            case init_conds::stationary:
                {
                    // no velocity
                    vector_field(rk_level, mesh_gid, 0) = 0.0;
                    vector_field(rk_level, mesh_gid, 1) = 0.0;
                    if (num_dims == 3) {
                        vector_field(rk_level, mesh_gid, 2) = 0.0;
                    }

                    break;
                }
            case init_conds::noICsVec:
                {
                    // nothing is done

                    break;
                }
            default:
                {
                    // nothing is done

                    break;
                }
        } // end of switch

    } // end loop over rk_num_bins


    // done setting the velocity
}  // end function paint_vector


/////////////////////////////////////////////////////////////////////////////
///
/// \fn paint_node_scalar
///
/// \brief a function to paint a scalars on the nodes of the mesh
///
/// \param The scalar value to be painted onto the nodes
/// \param Regions to fill
/// \param node_scalar is the nodal scalar array
/// \param node_coords are the coordinates of the nodes
/// \param node_gid is the element global mesh index
/// \param f_id is fill instruction
/// \param Number of dimensions of the mesh
/// \param The ID of the fill instruction
/// \param rk_num_bins is time integration storage level
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void paint_node_scalar(const double scalar,
                       const CArrayKokkos<RegionFill_t>& region_fills,
                       const DCArrayKokkos<double>& node_scalar,
                       const DCArrayKokkos<double>& node_coords,
                       const double node_gid,
                       const double num_dims,
                       const size_t f_id,
                       const size_t rk_num_bins)
{
    // save velocity at all rk_levels
    for(size_t rk_level = 0; rk_level < rk_num_bins; rk_level++){

        // --- scalar field ---
        switch (region_fills(f_id).temperature_field) {
            case init_conds::uniform:
                {

                    node_scalar(rk_level, node_gid) = scalar;
                    break;
                }
            // radial in the (x,y) plane where x=r*cos(theta) and y=r*sin(theta)
            case init_conds::radialScalar:
                {
                    // Setting up radial
                    double dir[2];
                    dir[0] = 0.0;
                    dir[1] = 0.0;
                    double radius_val = 0.0;

                    for (int dim = 0; dim < 2; dim++) {
                        dir[dim]    = node_coords(rk_level, node_gid, dim);
                        radius_val += node_coords(rk_level, node_gid, dim) * node_coords(rk_level, node_gid, dim);
                    } // end for
                    radius_val = sqrt(radius_val);

                    for (int dim = 0; dim < 2; dim++) {
                        if (radius_val > 1.0e-14) {
                            dir[dim] /= (radius_val);
                        }
                        else{
                            dir[dim] = 0.0;
                        }
                    } // end for

                    node_scalar(rk_level, node_gid) = scalar * dir[0];
                    node_scalar(rk_level, node_gid) = scalar * dir[1];

                    break;
                }
            case init_conds::sphericalScalar:
                {
                    // Setting up spherical
                    double dir[3];
                    dir[0] = 0.0;
                    dir[1] = 0.0;
                    dir[2] = 0.0;
                    double radius_val = 0.0;

                    for (int dim = 0; dim < 3; dim++) {
                        dir[dim]    = node_coords(rk_level, node_gid, dim);
                        radius_val += node_coords(rk_level, node_gid, dim) * node_coords(rk_level, node_gid, dim);
                    } // end for
                    radius_val = sqrt(radius_val);

                    for (int dim = 0; dim < 3; dim++) {
                        if (radius_val > 1.0e-14) {
                            dir[dim] /= (radius_val);
                        }
                        else{
                            dir[dim] = 0.0;
                        }
                    } // end for

                    node_scalar(rk_level, node_gid) = scalar * radius_val;
                    break;
                }
            case init_conds::tgVortexScalar:
                {
                    printf("**** TG Vortex not supported for general scalar initial conditions ****\n");

                    break;
                }
            case init_conds::noICsScalar:
                {
                    // nothing is done

                    break;
                }
            default:
                {
                    // do nothing

                    break;
                }
        } // end of switch

    } // end loop over rk_num_bins

}  // end function paint_node_scalar

/////////////////////////////////////////////////////////////////////////////
///
/// \fn paint_gauss_den_sie
///
/// \brief a function to paint den and sie on the Gauss points of the mesh 
///
/// \param Materials holds the material models and global parameters
/// \param mesh is the simulation mesh
/// \param node_coords are the node coordinates of the element
/// \param GaussPoint_den is density at the GaussPoints on the mesh
/// \param GaussPoint_sie is specific internal energy at the GaussPoints on the mesh
/// \param elem_mat_id is the material id in an element
/// \param region_fills are the instructions to paint state on the mesh
/// \param elem_coords is the geometric center of the element
/// \param elem_gid is the element global mesh index
/// \param f_id is fill instruction
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_FUNCTION
void paint_gauss_den_sie(const Material_t& Materials,
                         const Mesh_t& mesh,
                         const DCArrayKokkos <double>& node_coords,
                         const DCArrayKokkos <double>& GaussPoint_den,
                         const DCArrayKokkos <double>& GaussPoint_sie,
                         const DCArrayKokkos <double>& GaussPoint_volfrac,
                         const DCArrayKokkos <size_t>& elem_mat_id,
                         const DCArrayKokkos <size_t>& num_mats_saved_in_elem,
                         const CArrayKokkos<RegionFill_t>& region_fills,
                         const ViewCArrayKokkos <double> elem_coords,
                         const double elem_gid,
                         const size_t f_id)
{
    const double fuzzy_zero = 1.e-14;
    const double fuzzy_one = 1.0 - 1.e-14;

    // the number of materials saved to this element, initialized to 0 at start of code
    size_t mat_storage_lid = num_mats_saved_in_elem(elem_gid);


    // check on exceeding 3 materials per element
    if (num_mats_saved_in_elem(elem_gid) > 3){
        Kokkos::abort("ERROR: exceeded 3 materials in an element when painting regions on the mesh \n");
    } // end if check


    // material id
    const size_t mat_id = region_fills(f_id).material_id;


    // check to see if the material already exists
    bool check_mat_exists = false;
    for (size_t a_mat_in_elem=0; a_mat_in_elem < num_mats_saved_in_elem(elem_gid); a_mat_in_elem++){
        if(mat_id == elem_mat_id(elem_gid, a_mat_in_elem)){
            mat_storage_lid = a_mat_in_elem;  // set storage lid to the existing material lid
            check_mat_exists = true;
        } // end if check on mat_id existing alrady
    } // end for a_mat


    // There will now be at least 1 material so we want
    // num_mats_saved_in_elem >= 1, and it is intialized at 0 
    if(check_mat_exists == false){
        // we are adding a new material, so increment the number of saved
        num_mats_saved_in_elem(elem_gid) += 1;
    } // end check

    // check to see if the volfrac paint is equal to 1
    bool eraise_mats = false;
    size_t num_mats_previously_saved;
    if(region_fills(f_id).volfrac >= fuzzy_one){
        // this is used to wipe away materials later in this routine
        num_mats_previously_saved = num_mats_saved_in_elem(elem_gid);

        // if the volume fraction is >=1, reset the counter, deleting all prior materials
        num_mats_saved_in_elem(elem_gid) = 1;

        mat_storage_lid = 0; // save state to the zero index of the gauss point arrays
        eraise_mats = true;  
    } // end if volfrac >= 1



    // --- material_id in elem ---
    elem_mat_id(elem_gid, mat_storage_lid) = mat_id; // 


    // loop over the Gauss points in the element and save region initial conditions
    {
        
        const size_t gauss_gid = elem_gid;  // 1 gauss point per element


        // add test problem state setups here
        if (region_fills(f_id).vel_field == init_conds::tgVortexVec) {

            GaussPoint_den(gauss_gid, mat_storage_lid) = 1.0;    

            // note: elem_coords are the gauss_coords, higher quadrature requires ref elem data
            double pres = 0.25 * (cos(2.0 * PI * elem_coords(0)) + 
                                  cos(2.0 * PI * elem_coords(1)) ) + 1.0;

            // p = rho*ie*(gamma - 1)
            // makes sure index 0 matches the gamma in the gamma law function 
            double gamma  = Materials.eos_global_vars(mat_id,0); 
            GaussPoint_sie(gauss_gid, mat_storage_lid) =
                pres / (GaussPoint_den(gauss_gid, mat_storage_lid) * (gamma - 1.0));

            GaussPoint_volfrac(gauss_gid, mat_storage_lid) = region_fills(f_id).volfrac;
        } // end
        // *****
        // add user initialization here
        // *****
        else {
            
            // --- density ---
            GaussPoint_den(gauss_gid, mat_storage_lid) = region_fills(f_id).den;

            // --- specific internal energy ---
            GaussPoint_sie(gauss_gid, mat_storage_lid) = region_fills(f_id).sie;

            // --- volume fraction ---
            GaussPoint_volfrac(gauss_gid, mat_storage_lid) = region_fills(f_id).volfrac;

        }  // end if 

        //printf("volfrac in elem = %f \n",GaussPoint_volfrac(gauss_gid, mat_storage_lid));

        if(eraise_mats == true){
            // remove materials after mat_stroage_lid=0 by seting volfrac=0
            for (size_t a_mat_in_elem=1; a_mat_in_elem < num_mats_previously_saved; a_mat_in_elem++){
                GaussPoint_volfrac(gauss_gid, a_mat_in_elem) = 0.0;
            }
        } // end check on eraising mats

        //printf("volfrac in elem after eraise = %f \n",GaussPoint_volfrac(gauss_gid, mat_storage_lid));



        // ----- 
        // ensure volume fraction is bounded between 0 and 1 for all materials
        // ----- 
        double vol_frac_total = 0.0;
        for (size_t a_mat_in_elem=0; a_mat_in_elem < num_mats_saved_in_elem(elem_gid); a_mat_in_elem++){
            vol_frac_total += GaussPoint_volfrac(gauss_gid, a_mat_in_elem);
        }

        //printf("volfrac total = %f \n",vol_frac_total);
        
        // squish material out if vol fraction in element is > 1
        if (vol_frac_total > 1.0){

            double vol_error = vol_frac_total - 1.0;

            // squish out material
            for (size_t a_mat_in_elem=0; a_mat_in_elem < num_mats_saved_in_elem(elem_gid); a_mat_in_elem++){
                double vol_removed = fmin(GaussPoint_volfrac(gauss_gid, a_mat_in_elem), vol_error);
                GaussPoint_volfrac(gauss_gid, a_mat_in_elem) -= vol_removed;
                vol_error -= fmax(0.0, vol_removed);  // once error =0, no more material removed
            } // end for squishing out material

            // verify that every volume fraction is bounded 0:1
            for (size_t a_mat_in_elem=0; a_mat_in_elem < num_mats_saved_in_elem(elem_gid); a_mat_in_elem++){
                double volfracfloor = fmax(0.0, GaussPoint_volfrac(gauss_gid, a_mat_in_elem));
                GaussPoint_volfrac(gauss_gid, a_mat_in_elem) = fmin(1.0, volfracfloor);
            } // end loop on bounded volfracs

            //for (size_t a_mat_in_elem=0; a_mat_in_elem < num_mats_saved_in_elem(elem_gid); a_mat_in_elem++){
            //    printf("volfrac in elem after squishing = %f \n",GaussPoint_volfrac(gauss_gid, a_mat_in_elem));
            //}

        } // end if too much material in an element

         //printf("volfrac in elem after squishing = %f \n",GaussPoint_volfrac(gauss_gid, mat_storage_lid));

        // ----- 
        // remove all materials that have a zero vol fraction
        // ----- 
        size_t num_actual_mats = 0;
        for (size_t a_mat_in_elem=0; a_mat_in_elem < num_mats_saved_in_elem(elem_gid); a_mat_in_elem++){
            
            // if volfraction is greater than zero, them material is hear
             if( GaussPoint_volfrac(gauss_gid, a_mat_in_elem) >= fuzzy_zero ){
                num_actual_mats++;
             } // end if

        } // end loop over mats in this elem

        //printf("num actual mats = %zu \n",num_actual_mats);


        size_t shift = num_mats_saved_in_elem(elem_gid)-num_actual_mats;
        num_mats_saved_in_elem(elem_gid) = num_actual_mats;
        for (size_t a_mat_in_elem=0; a_mat_in_elem < num_actual_mats; a_mat_in_elem++){

            // if nothing is at this gauss point, remove it, compressing out the zero volfracs
            if( GaussPoint_volfrac(gauss_gid, a_mat_in_elem) <= fuzzy_zero ){
                
                // move density up in the storage
                GaussPoint_den(gauss_gid, a_mat_in_elem) = GaussPoint_den(gauss_gid, a_mat_in_elem+shift);  // material density

                // move specific internal energy up in the storage
                GaussPoint_sie(gauss_gid, a_mat_in_elem) = GaussPoint_sie(gauss_gid, a_mat_in_elem+shift);  // material sie

                // move volume fraction up in the storage
                GaussPoint_volfrac(gauss_gid, a_mat_in_elem) = GaussPoint_volfrac(gauss_gid, a_mat_in_elem+shift);

            }; // end check on volfrac being zero

        } // end loop over materials



       // printf("volfrac in elem end = %f \n",GaussPoint_volfrac(gauss_gid, mat_storage_lid));

        

    } // end loop over gauss points in element

    // done setting the element state


} // end function

/////////////////////////////////////////////////////////////////////////////
///
/// \fn init_state_vars
///
/// \brief a function to initialize eos and stress state vars
///
/// \param Materials holds the material models and global parameters
/// \param mesh is the simulation mesh
/// \param DualArrays for the material point eos state vars
/// \param DualArrays for the material point strength state vars
/// \param rk_num_bins is number of time integration storage bins
/// \param num_mat_pts is the number of material points for mat_id
/// \param mat_id is material id
///
/////////////////////////////////////////////////////////////////////////////
void init_state_vars(const Material_t& Materials,
                     const Mesh_t& mesh,
                     const DCArrayKokkos<double>& MaterialPoints_eos_state_vars,
                     const DCArrayKokkos<double>& MaterialPoints_strength_state_vars,
                     const DCArrayKokkos<size_t>& MaterialToMeshMaps_elem,
                     const size_t rk_num_bins,
                     const size_t num_mat_pts,
                     const size_t mat_id)
{

    // -------
    // the call to the model initialization
    // -------
    if (Materials.MaterialEnums.host(mat_id).StrengthType == model::incrementBased ||
        Materials.MaterialEnums.host(mat_id).StrengthType == model::stateBased) {

            if (Materials.MaterialEnums.host(mat_id).StrengthSetupLocation == model::host){

                Materials.MaterialFunctions.host(mat_id).init_strength_state_vars(
                                MaterialPoints_eos_state_vars,
                                MaterialPoints_strength_state_vars,
                                Materials.eos_global_vars,
                                Materials.strength_global_vars,
                                MaterialToMeshMaps_elem,
                                num_mat_pts,
                                mat_id);

            } // end if
            else {
                // --- running setup function on the device

                printf("Calling initial condition function on GPU is NOT yet supported \n");

            }

    } // end if

} // end of set values in eos and strength state vars



/////////////////////////////////////////////////////////////////////////////
///
/// \fn init_press_sspd_stress
///
/// \brief a function to initialize pressure, sound speed and stress
///
/// \param Materials holds the material models and global parameters
/// \param mesh is the simulation mesh
/// \param DualArrays for density at the material points on the mesh
/// \param DualArrays for pressure at the material points on the mesh
/// \param DualArrays for stress at the material points on the mesh
/// \param DualArrays for sound speed at the material points on the mesh
/// \param DualArrays for specific internal energy at the material points on the mesh
/// \param DualArrays for the material point eos state vars
/// \param DualArrays for the material point strength state vars
/// \param num_mat_pts is the number of material points for mat_id
/// \param mat_id is material id
/// \param rk_num_bins is number of time integration storage bins
///
/////////////////////////////////////////////////////////////////////////////
void init_press_sspd_stress(const Material_t& Materials,
                            const Mesh_t& mesh,
                            const DCArrayKokkos<double>& MaterialPoints_den,
                            const DCArrayKokkos<double>& MaterialPoints_pres,
                            const DCArrayKokkos<double>& MaterialPoints_stress,
                            const DCArrayKokkos<double>& MaterialPoints_sspd,
                            const DCArrayKokkos<double>& MaterialPoints_sie,
                            const DCArrayKokkos<double>& MaterialPoints_eos_state_vars,
                            const DCArrayKokkos<double>& MaterialPoints_strength_state_vars,
                            const DCArrayKokkos<double>& MaterialPoints_shear_modulii,
                            const size_t rk_num_bins,
                            const size_t num_mat_pts,
                            const size_t mat_id)
{
    std::cout << "Before setting shear modulus to zero" << std::endl;
    // --- Shear modulus ---
    // loop over the material points

    if (MaterialPoints_shear_modulii.size() > 0) {
        FOR_ALL(mat_point_lid, 0, num_mat_pts, {

            // setting shear modulii to zero, corresponds to a gas
            for(size_t i; i<3; i++){
                MaterialPoints_shear_modulii(mat_point_lid,i) = 0.0;
            } // end for

        });
    }

    
    // --- stress tensor ---
    for(size_t rk_level=0; rk_level<rk_num_bins; rk_level++){                

        FOR_ALL(mat_point_lid, 0, num_mat_pts, {

            // always 3D even for 2D-RZ
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {

                    // ===============
                    //  Call the strength model here
                    // ===============
                    MaterialPoints_stress(rk_level, mat_point_lid, i, j) = 0.0;
                }
            }  // end for i,j
                             
        }); // end parallel for over matpt storage

    }// end for rk_level


    // --- pressure and sound speed ---
    // loop over the material points
    FOR_ALL(mat_point_lid, 0, num_mat_pts, {

        // --- Pressure ---
        Materials.MaterialFunctions(mat_id).calc_pressure(
                                        MaterialPoints_pres,
                                        MaterialPoints_stress,
                                        mat_point_lid,
                                        mat_id,
                                        MaterialPoints_eos_state_vars,
                                        MaterialPoints_sspd,
                                        MaterialPoints_den(mat_point_lid),
                                        MaterialPoints_sie(0, mat_point_lid),
                                        Materials.eos_global_vars);   

        // --- Sound Speed ---                               
        Materials.MaterialFunctions(mat_id).calc_sound_speed(
                                        MaterialPoints_pres,
                                        MaterialPoints_stress,
                                        mat_point_lid,
                                        mat_id,
                                        MaterialPoints_eos_state_vars,
                                        MaterialPoints_sspd,
                                        MaterialPoints_den(mat_point_lid),
                                        MaterialPoints_sie(0, mat_point_lid),
                                        MaterialPoints_shear_modulii,
                                        Materials.eos_global_vars);
    }); // end pressure and sound speed




} // end function



/////////////////////////////////////////////////////////////////////////////
///
/// \fn calc_corner_mass
///
/// \brief a function to initialize pressure, sound speed and stress
///
/// \param Materials holds the material models and global parameters
/// \param mesh is the simulation mesh
/// \param node_coords are the nodal coordinates of the mesh
/// \param node_mass is mass of the node
/// \param corner_mass is corner mass
/// \param MaterialPoints_mass is the mass at the material point for mat_id
/// \param num_mat_elems is the number of material elements for mat_id
///
/////////////////////////////////////////////////////////////////////////////
void calc_corner_mass(const Material_t& Materials,
                      const Mesh_t& mesh,
                      const DCArrayKokkos<double>& node_coords,
                      const DCArrayKokkos<double>& node_mass,
                      const DCArrayKokkos<double>& corner_mass,
                      const DCArrayKokkos<double>& MaterialPoints_mass,
                      const DCArrayKokkos<size_t>& MaterialToMeshMaps_elem,
                      const size_t num_mat_elems)
{


    FOR_ALL(mat_elem_lid, 0, num_mat_elems, {

        // get elem gid
        size_t elem_gid = MaterialToMeshMaps_elem(mat_elem_lid);  

        // calculate the fraction of matpt mass to scatter to each corner
        double corner_frac = 1.0/((double)mesh.num_nodes_in_elem);  // =1/8
        
        // partion the mass to the corners
        for(size_t corner_lid=0; corner_lid<mesh.num_nodes_in_elem; corner_lid++){
            size_t corner_gid = mesh.corners_in_elem(elem_gid, corner_lid);
            corner_mass(corner_gid) += corner_frac*MaterialPoints_mass(mat_elem_lid);
        } // end for

    }); // end parallel for over mat elem local ids


} // end function calculate corner mass


/////////////////////////////////////////////////////////////////////////////
///
/// \fn calc_node_mass
///
/// \brief a function to initialize material corner masses
///
/// \param Materials holds the material models and global parameters
/// \param mesh is the simulation mesh
/// \param node_coords are the nodal coordinates of the mesh
/// \param node_mass is mass of the node
/// \param corner_mass is corner mass
/// \param MaterialPoints_mass is the mass at the material point for mat_id
/// \param num_mat_elems is the number of material elements for mat_id
///
/////////////////////////////////////////////////////////////////////////////
void calc_node_mass(const Mesh_t& mesh,
                    const DCArrayKokkos<double>& node_coords,
                    const DCArrayKokkos<double>& node_mass,
                    const DCArrayKokkos<double>& corner_mass)
{


    FOR_ALL(node_gid, 0, mesh.num_nodes, {
        for (size_t corner_lid = 0; corner_lid < mesh.num_corners_in_node(node_gid); corner_lid++) {

            size_t corner_gid = mesh.corners_in_node(node_gid, corner_lid);

            node_mass(node_gid) += corner_mass(corner_gid);
        } // end for elem_lid
    }); // end parallel loop over nodes in the mesh

} // end function calculate node mass
