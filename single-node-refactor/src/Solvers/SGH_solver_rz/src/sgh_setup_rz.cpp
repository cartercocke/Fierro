/**********************************************************************************************
© 2020. Triad National Security, LLC. All rights reserved.
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

#include "sgh_solver_rz.h"
#include "mesh.h"
#include "region_fill.h"
#include "material.h"
#include "boundary_conditions.h"
#include "simulation_parameters.h"
#include "state.h"
#include "geometry_new.h"

/////////////////////////////////////////////////////////////////////////////
///
/// \fn init_corner_node_masses_zero
///
/// \brief a function to initialize corner and node masses to zero
///
/// \param mesh is the simulation mesh
/// \param node_mass is the node mass
/// \param corner_mass is the corner mass
///
/////////////////////////////////////////////////////////////////////////////
void SGHRZ::init_corner_node_masses_zero_rz(const Mesh_t& mesh,
                                            const DCArrayKokkos<double>& node_mass,
                                            const DCArrayKokkos<double>& corner_mass) const
{
                    
    // calculate the nodal mass
    FOR_ALL(node_gid, 0, mesh.num_nodes, {
        node_mass(node_gid) = 0.0;
    }); // end parallel over nodes

    FOR_ALL(corner_gid, 0, mesh.num_corners, {
        corner_mass(corner_gid) = 0.0;
    });  // end parallel over corners

} // end setting masses equal to zero

/////////////////////////////////////////////////////////////////////////////
///
/// \fn fill_regions_sgh_rz
///
/// \brief a function to paint den, sie, vel, and mat_ids on the mesh 
/// The arrays populated (on host and device) are:
///       elem_mat_id
///       GaussPoint_den
///       GaussPoint_sie
///       node_vel
///
/// \param Materials holds the material models and global parameters
/// \param mesh is the simulation mesh
/// \param node_coords are the coordinates of the nodes
/// \param node_vel is the nodal velocity array
/// \param GaussPoint_den is density at the GaussPoints on the mesh
/// \param GaussPoint_sie is specific internal energy at the GaussPoints on the mesh
/// \param GaussPoint_volfrac is volume fraction at the GaussPoints on the mesh
/// \param elem_mat_id is the material id in an element
/// \param num_mats_saved_in_elem is the number of material with volfrac<1 saved to the element
/// \param voxel_elem_mat_id are the voxel values on a structured i,j,k mesh
/// \param object_ids are the object ids in the vtu file
/// \param reg_fills_in_solver are the regions to fill for this solver
/// \param region_fills are the instructures to paint state on the mesh
/// \param region_fills_host are the instructures on the host side to paint state on the mesh
/// \param num_fills_in_solver is number of fill instruction for the solver
/// \param num_elems is number of elements on the mesh
/// \param num_nodes is number of nodes on the mesh
/// \param rk_num_bins is number of time integration storage bins
///
/////////////////////////////////////////////////////////////////////////////
void SGHRZ::fill_regions_sgh_rz(
                    const Material_t& Materials,
                    const Mesh_t& mesh,
                    const DCArrayKokkos <double>& node_coords,
                    DCArrayKokkos <double>& node_vel,
                    DCArrayKokkos <double>& GaussPoint_den,
                    DCArrayKokkos <double>& GaussPoint_sie,
                    DCArrayKokkos <double>& GaussPoint_volfrac,
                    DCArrayKokkos <size_t>& elem_mat_id,
                    DCArrayKokkos <size_t>& num_mats_saved_in_elem,
                    DCArrayKokkos <size_t>& voxel_elem_mat_id,
                    const DCArrayKokkos <int>& object_ids,
                    const DCArrayKokkos<size_t>& reg_fills_in_solver,
                    const CArrayKokkos <RegionFill_t>& region_fills,
                    const CArray <RegionFill_host_t>& region_fills_host,
                    const size_t num_fills_in_solver,
                    const size_t rk_num_bins) const
{
    double voxel_dx, voxel_dy, voxel_dz;          // voxel mesh resolution, set by input file
    double orig_x, orig_y, orig_z;                // origin of voxel elem center mesh, set by input file
    size_t voxel_num_i, voxel_num_j, voxel_num_k; // num voxel elements in each direction, set by input file

    size_t num_fills_total = region_fills.size();  // the total number of fills in the input file

    // ---------------------------------------------
    // copy to host, enum to read a voxel file
    // ---------------------------------------------
    
    DCArrayKokkos<size_t> read_voxel_file(num_fills_total); // check to see if readVoxelFile

    FOR_ALL(fill_id, 0, num_fills_total, {
        if (region_fills(fill_id).volume == region::readVoxelFile)
        {
            read_voxel_file(fill_id) = region::readVoxelFile;  // read the  voxel file
        }
        // add other mesh voxel files
        else
        {
            read_voxel_file(fill_id) = 0;
        }
    }); // end parallel for
    read_voxel_file.update_host(); // copy to CPU if code is to read a file
    Kokkos::fence();
    // ---------------------------------------------


    // loop over the fill instructions for this solver
    for (size_t f_lid = 0; f_lid < num_fills_in_solver; f_lid++) {

        // get the fill id
        size_t fill_id = reg_fills_in_solver.host(this->solver_id, f_lid);

        // ----
        // voxel mesh setup
        if (read_voxel_file.host(fill_id) == region::readVoxelFile)
        {
            // read voxel mesh to get the values in the fcn interface
            user_voxel_init(voxel_elem_mat_id,
                            voxel_dx, 
                            voxel_dy, 
                            voxel_dz,
                            orig_x, 
                            orig_y, 
                            orig_z,
                            voxel_num_i, 
                            voxel_num_j, 
                            voxel_num_k,
                            region_fills_host(fill_id).scale_x,
                            region_fills_host(fill_id).scale_y,
                            region_fills_host(fill_id).scale_z,
                            region_fills_host(fill_id).file_path);

            // copy values read from file to device
            voxel_elem_mat_id.update_device();
        } // endif
        // add else if for other mesh reads including STL-2-voxel


        // parallel loop over elements in mesh
        FOR_ALL(elem_gid, 0, mesh.num_elems, {

            // calculate the coordinates and radius of the element
            double elem_coords_1D[3]; // note:initialization with a list won't work
            ViewCArrayKokkos <double> elem_coords(&elem_coords_1D[0], 3);
            elem_coords(0) = 0.0;
            elem_coords(1) = 0.0;
            elem_coords(2) = 0.0;

            // get the coordinates of the element center (using rk_level=1 or node coords)
            for (int node_lid = 0; node_lid < mesh.num_nodes_in_elem; node_lid++) {
                elem_coords(0) += node_coords(1, mesh.nodes_in_elem(elem_gid, node_lid), 0);
                elem_coords(1) += node_coords(1, mesh.nodes_in_elem(elem_gid, node_lid), 1);


            } // end loop over nodes in element 
            elem_coords(0) = (elem_coords(0) / mesh.num_nodes_in_elem);
            elem_coords(1) = (elem_coords(1) / mesh.num_nodes_in_elem);
            

            
            // calc if we are to fill this element
            size_t fill_this = fill_geometric_region(mesh, 
                                                     voxel_elem_mat_id, 
                                                     object_ids,
                                                     region_fills, 
                                                     elem_coords, 
                                                     voxel_dx, 
                                                     voxel_dy, 
                                                     voxel_dz,
                                                     orig_x, 
                                                     orig_y, 
                                                     orig_z,
                                                     voxel_num_i, 
                                                     voxel_num_j, 
                                                     voxel_num_k,
                                                     fill_id,
                                                     elem_gid);


            // paint the material state on the element if fill_this=1
            if (fill_this == 1) {

                // default sgh_rz paint
                paint_gauss_den_sie(Materials,
                                    mesh,
                                    node_coords,
                                    GaussPoint_den,
                                    GaussPoint_sie,
                                    GaussPoint_volfrac,
                                    elem_mat_id,
                                    num_mats_saved_in_elem,
                                    region_fills,
                                    elem_coords,
                                    elem_gid,
                                    fill_id);

                // add user defined paint here
                // user_defined_sgh_state();


                // technically, not thread safe, but making it a separate loop created bad fill behavior
                // loop over the nodes of this element and apply velocity
                for (size_t node_lid = 0; node_lid < mesh.num_nodes_in_elem; node_lid++) {
                        
                    // get the mesh node index
                    size_t node_gid = mesh.nodes_in_elem(elem_gid, node_lid);       

                    // node coords(rk,node_gid,dim), using the first rk level in the view
                    ViewCArrayKokkos <double> a_node_coords(&node_coords(0,node_gid,0), 3);
                    
                    // default sghRZ paint
                    paint_vector_rk(node_vel,
                                a_node_coords,
                                region_fills(fill_id).u,
                                region_fills(fill_id).v,
                                region_fills(fill_id).w,
                                region_fills(fill_id).speed,
                                node_gid,
                                mesh.num_dims,
                                rk_num_bins,
                                region_fills(fill_id).vel_field);

                    // add user defined paint here
                    // user_defined_vel_state();
                    
                } // end loop over the nodes in elem

            } // end if fill this

        }); // end FOR_ALL node loop
        Kokkos::fence();

    } // end for loop over fills


    elem_mat_id.update_host();
    GaussPoint_den.update_host();
    GaussPoint_sie.update_host();
    GaussPoint_volfrac.update_host();
    node_vel.update_host();
    num_mats_saved_in_elem.update_host();

    Kokkos::fence();

} // end SGHRZ fill regions


/////////////////////////////////////////////////////////////////////////////
///
/// \fn setup the SGHRZ method
///
/// \brief Allocate state, setup models, and fill mesh regions per the YAML input
///
/////////////////////////////////////////////////////////////////////////////
void SGHRZ::setup(SimulationParameters_t& SimulationParamaters, 
                Material_t& Materials, 
                Mesh_t& mesh, 
                BoundaryCondition_t& Boundary,
                State_t& State)
{

    size_t num_fills_in_solver = SimulationParamaters.region_setups.num_reg_fills_in_solver.host(this->solver_id);
    printf("Num fills's = %zu\n in solver = %zu", num_fills_in_solver, this->solver_id);

    // the number of elems and nodes in the mesh
    const size_t num_elems = mesh.num_elems;
    const size_t num_nodes = mesh.num_nodes;
    const size_t num_gauss_points = mesh.num_elems;  // 1 Gauss point per element

    const size_t rk_num_bins = SimulationParamaters.dynamic_options.rk_num_bins;

    // Calculate element volume
    geometry::get_vol(State.GaussPoints.vol, State.node.coords, mesh);


    // create temporary state fields
    // Painting routine requires only 1 material per GaussPoint
    // allowing for up to 3 materials in an element
    const size_t num_mats_per_elem = 3;
    DCArrayKokkos <double> GaussPoint_den(num_gauss_points, num_mats_per_elem, "GaussPoint_den");
    DCArrayKokkos <double> GaussPoint_sie(num_gauss_points, num_mats_per_elem, "GaussPoint_sie");
    DCArrayKokkos <double> GaussPoint_volfrac(num_gauss_points, num_mats_per_elem, "GaussPoint_vofrac");
    DCArrayKokkos <size_t> elem_mat_id(num_elems, num_mats_per_elem, "elem_mat_id"); // the mat_id in the elem

    // num mats saved in an element during setup
    DCArrayKokkos <size_t> num_mats_saved_in_elem(num_elems, "num_mats_saved_in_elem");
    num_mats_saved_in_elem.set_values(0); // initialize all elems to storing 0 materials
    num_mats_saved_in_elem.update_host();

    DCArrayKokkos<size_t> voxel_elem_mat_id;       // 1 or 0 if material exist, or it is the material_id


    // Verify all node radii are positive
    for(int node_gid = 0; node_gid < num_nodes; node_gid++){
        if(State.node.coords.host(0, node_gid, 0) < 0.0){
            throw std::runtime_error("**** NODE RADIUS FOR RZ MESH MUST BE POSITIVE ****");
        }
    }


    // ---------------------------------------------
    // fill den, sie, and velocity on the mesh
    // ---------------------------------------------
    fill_regions_sgh_rz(Materials,
                        mesh,
                        State.node.coords,
                        State.node.vel,
                        GaussPoint_den,
                        GaussPoint_sie,
                        GaussPoint_volfrac,
                        elem_mat_id,
                        num_mats_saved_in_elem,
                        voxel_elem_mat_id,
                        SimulationParamaters.mesh_input.object_ids,
                        SimulationParamaters.region_setups.reg_fills_in_solver,
                        SimulationParamaters.region_setups.region_fills,
                        SimulationParamaters.region_setups.region_fills_host,
                        num_fills_in_solver,
                        rk_num_bins);


    // note: the device and host side are updated in the above function
    // ---------------------------------------------

    
    // ----------------------------------------------------------------
    //  Walk over the mesh and find dimensions of material storage arrays
    // ----------------------------------------------------------------
    const size_t num_mats = Materials.num_mats; // the number of materials on the mesh

    // a counter for the Material index spaces
    DCArrayKokkos <size_t> num_elems_saved_for_mat(num_mats, "num_elems_saved_for_mat");  

    for(int mat_id=0; mat_id<num_mats; mat_id++){

        size_t sum_local;
        size_t sum_total;

        FOR_REDUCE_SUM(elem_gid, 0, num_elems, sum_local,{

            // loop over the materials in the element
            for (size_t a_mat_in_elem=0; a_mat_in_elem < num_mats_saved_in_elem(elem_gid); a_mat_in_elem++){

                if(elem_mat_id(elem_gid, a_mat_in_elem) == mat_id){
                    // increment the number of elements the materials live in
                    sum_local++;
                } // end if    
            } // end loop over materials in elem

        }, sum_total);

        // material index space size
        num_elems_saved_for_mat.host(mat_id) = sum_total;

    } // end for



    // ---------------------------------------
    //  SGHRZ allocation of maps and state
    // ---------------------------------------
    State.MaterialToMeshMaps = CArray<MaterialToMeshMap_t> (num_mats);

    State.MaterialPoints  = CArray<MaterialPoint_t> (num_mats);
    State.MaterialCorners = CArray<MaterialCorner_t> (num_mats);
    // zones not needed with SGHRZ
    
    
    // for ALE SGHRZ, add a buffer to num_elems_for_mat, like 10% of num_elems up to num_elems.
    for(int mat_id=0; mat_id<num_mats; mat_id++){

        const size_t num_mat_pts_in_elem = 1; // 1 mat_point per elem with SGHRZ

        // the following always have the exact memory needed, they omit the buffer
        State.MaterialToMeshMaps(mat_id).num_material_elems = num_elems_saved_for_mat.host(mat_id); 
        State.MaterialPoints(mat_id).num_material_points = num_elems_saved_for_mat.host(mat_id)*num_mat_pts_in_elem;
        State.MaterialCorners(mat_id).num_material_corners = num_elems_saved_for_mat.host(mat_id)*mesh.num_nodes_in_elem;

        // -----
        //  Allocation after here will include a buffer
        // -----
        size_t buffer = 0; // memory buffer to push back into
        size_t num_elems_for_mat = num_elems_saved_for_mat.host(mat_id)+buffer; // has a memory buffer for ALE

        size_t num_points_for_mat = num_elems_for_mat*num_mat_pts_in_elem;  
        size_t num_corners_for_mat = num_elems_for_mat*mesh.num_nodes_in_elem;

        State.MaterialToMeshMaps(mat_id).initialize(num_elems_for_mat); 
        State.MaterialPoints(mat_id).initialize(rk_num_bins, num_points_for_mat, 3, SGHRZ_State::required_material_pt_state); // aways 3D, even for 2D-RZ calcs
        State.MaterialCorners(mat_id).initialize(num_corners_for_mat, mesh.num_dims, SGHRZ_State::required_material_corner_state); 
        // zones are not used
    
    } // end for mat_id
    
    // data structures to access indices in other material index spaces
    State.corners_in_mat_elem = corners_in_mat_t(mesh.num_nodes_in_elem); 
    State.points_in_mat_elem  = points_in_mat_t(1);  // 1 material point per element
    // zones_in_mat_elem is not used with SGHRZ



    // now a counter for how many elems have been saved
    for(int mat_id=0; mat_id<num_mats; mat_id++){
        num_elems_saved_for_mat.host(mat_id) = 0; // initializing to zero
    }



    // ---------------------------------------
    //  SGHRZ save data, maps, and state
    // ---------------------------------------
    State.GaussPoints.vol.update_host(); 
    Kokkos::fence();

    // the following loop is not thread safe
    for(size_t elem_gid=0; elem_gid<num_elems; elem_gid++){
        for (size_t a_mat_in_elem=0; a_mat_in_elem < num_mats_saved_in_elem.host(elem_gid); a_mat_in_elem++){


            // get the material_id in this element
            size_t mat_id = elem_mat_id.host(elem_gid,a_mat_in_elem);

            // mat elem lid (compressed storage) to save the data to, for this material mat_id
            size_t mat_elem_lid = num_elems_saved_for_mat.host(mat_id); 

            // --- mapping from material elem lid to elem ---
            State.MaterialToMeshMaps(mat_id).elem.host(mat_elem_lid) = elem_gid;

            // -----------------------
            // Save MaterialPoints
            // -----------------------

            // LOOP OVER Guass points in the element
            {
                size_t gauss_gid = elem_gid;  // 1 gauss point per element

                size_t mat_point_lid = mat_elem_lid; // for more than 1 gauss point, this must increment

                // --- density and mass ---
                State.MaterialPoints(mat_id).den.host(mat_point_lid)  = GaussPoint_den.host(gauss_gid,a_mat_in_elem); 
                State.MaterialPoints(mat_id).mass.host(mat_point_lid) = GaussPoint_den.host(gauss_gid,a_mat_in_elem) * 
                                                                        State.GaussPoints.vol.host(gauss_gid) *
                                                                        GaussPoint_volfrac.host(gauss_gid,a_mat_in_elem);

                // --- volume fraction ---
                State.MaterialPoints(mat_id).volfrac.host(mat_point_lid) = GaussPoint_volfrac.host(gauss_gid,a_mat_in_elem);

                // --- set eroded flag to false ---
                State.MaterialPoints(mat_id).eroded.host(mat_point_lid) = false;

                // --- specific internal energy ---
                // save state, that is integrated in time, at the RK levels
                for(size_t rk_level=0; rk_level<rk_num_bins; rk_level++){
                    State.MaterialPoints(mat_id).sie.host(rk_level, mat_point_lid) = GaussPoint_sie.host(gauss_gid,a_mat_in_elem);
                }
            } // end loop over gauss points in element
        

            // -----------------------
            // Save MaterialZones
            // -----------------------
            // For higher-order FE, least squares fit the sie at gauss points to get zone values

            
            // update counter for how many mat_elem_lid values have been saved
            num_elems_saved_for_mat.host(mat_id)++;
        } // end loop over materials in this element
    } // end serial for loop over all elements

    // copy the state to the device
    for(int mat_id=0; mat_id<num_mats; mat_id++){
        State.MaterialPoints(mat_id).den.update_device();
        State.MaterialPoints(mat_id).mass.update_device();
        State.MaterialPoints(mat_id).sie.update_device();

        State.MaterialPoints(mat_id).volfrac.update_device();
        State.MaterialPoints(mat_id).eroded.update_device();

        State.MaterialToMeshMaps(mat_id).elem.update_device();
    } // end for
    Kokkos::fence();


    // calculate pressure, sound speed, and stress for each material
    for(int mat_id=0; mat_id<num_mats; mat_id++){

        size_t num_mat_points = State.MaterialPoints(mat_id).num_material_points;

        init_press_sspd_stress(Materials,
                            mesh,
                            State.MaterialPoints(mat_id).den,
                            State.MaterialPoints(mat_id).pres,
                            State.MaterialPoints(mat_id).stress,
                            State.MaterialPoints(mat_id).sspd,
                            State.MaterialPoints(mat_id).sie,
                            State.MaterialPoints(mat_id).eos_state_vars,
                            State.MaterialPoints(mat_id).strength_state_vars,
                            State.MaterialPoints(mat_id).shear_modulii,
                            rk_num_bins,
                            num_mat_points,
                            mat_id);

    } // for loop over mat_id


    // set corner and node masses to zero
    init_corner_node_masses_zero_rz(mesh, State.node.mass, State.corner.mass);



    // 2D RZ
    // calculate the corner massess if 2D

    for(int mat_id=0; mat_id<num_mats; mat_id++){

        size_t num_mat_elems = State.MaterialToMeshMaps(mat_id).num_material_elems;
        
        calc_corner_mass_rz(Materials,
                            mesh,
                            State.node.coords,
                            State.node.mass,
                            State.corner.mass,
                            State.MaterialPoints(mat_id).den,
                            State.MaterialToMeshMaps(mat_id).elem,
                            num_mat_elems);
    } // end for mat_id

    calc_node_mass_rz(mesh,
                      State.node.coords,
                      State.node.mass,
                      State.corner.mass);

} // end SGHRZ setup


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
void calc_corner_mass_rz(const Material_t& Materials,
                         const Mesh_t& mesh,
                         const DCArrayKokkos<double>& node_coords,
                         const DCArrayKokkos<double>& node_mass,
                         const DCArrayKokkos<double>& corner_mass,
                         const DCArrayKokkos<double>& MaterialPoints_den,
                         const DCArrayKokkos<size_t>& MaterialToMeshMaps_elem,
                         const size_t num_mat_elems)
{

    FOR_ALL(mat_elem_lid, 0, num_mat_elems, {

        // get elem gid
        size_t elem_gid = MaterialToMeshMaps_elem(mat_elem_lid); 

        // facial area of the corners
        double corner_areas_array[4];

        ViewCArrayKokkos<double> corner_areas(&corner_areas_array[0], 4);
        ViewCArrayKokkos<size_t> elem_node_gids(&mesh.nodes_in_elem(elem_gid, 0), 4);

        geometry::get_area_weights2D(corner_areas, elem_gid, node_coords, elem_node_gids);

        // loop over the corners of the element and calculate the mass
        for (size_t corner_lid = 0; corner_lid < 4; corner_lid++) {
            size_t corner_gid = mesh.corners_in_elem(elem_gid, corner_lid);
            corner_mass(corner_gid) += corner_areas(corner_lid) * MaterialPoints_den(mat_elem_lid); // node radius is added later
        } // end for over corners
    });

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
void calc_node_mass_rz(const Mesh_t& mesh,
                    const DCArrayKokkos<double>& node_coords,
                    const DCArrayKokkos<double>& node_mass,
                    const DCArrayKokkos<double>& corner_mass)
{

    FOR_ALL(node_gid, 0, mesh.num_nodes, {
        for (size_t corner_lid = 0; corner_lid < mesh.num_corners_in_node(node_gid); corner_lid++) {
            
            size_t corner_gid    = mesh.corners_in_node(node_gid, corner_lid);

            node_mass(node_gid) += corner_mass(corner_gid);  // sans the radius so it is areal node mass

            corner_mass(corner_gid) *= node_coords(1, node_gid, 1); // true corner mass now
        } // end for elem_lid
    });

} // end function calculate node mass













