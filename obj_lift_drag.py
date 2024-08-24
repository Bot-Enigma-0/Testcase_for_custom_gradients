import vtk, os, numpy as np, jax, jax.numpy as jnp, vtk.util.numpy_support as VN, time, pandas as pd, matplotlib.pyplot as plt

jax.devices('cpu')
jax.config.update('jax_enable_x64', True)
np.set_printoptions(precision=14)
pd.set_option('display.precision', 14)

# Set script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

start = time.time()

def read_vtu(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    return reader, data

def extract_data(data, array_names):
    points = VN.vtk_to_numpy(data.GetPoints().GetData())
    arrays = [jnp.array(VN.vtk_to_numpy(data.GetPointData().GetArray(name))) for name in array_names]
    return points, arrays

def add_data_to_vtk(data, gradient, field_names):
    for i, field_name in enumerate(field_names):
        gradient_array = VN.numpy_to_vtk(gradient[:, i], deep=True)
        gradient_array.SetName(field_name)
        data.GetPointData().AddArray(gradient_array)

def write_vtu(data, filename):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Write()

@jax.jit
def calc_lift(U, x, y, nodeID):
    lift = 0
    wedge_x, wedge_y = x[nodeID], y[nodeID]
    
    # Import the conservative variables --> all are dimensional
    normaliser = 0.5 * U[0,0] * (U[0,1]/U[0,0])**2 # --> freestream
    
    # calculate lift over the nodes
    gamma = 1.4
    for i in range(len(nodeID)):
        # Get local conservative variables for each node
        rho = U[nodeID[i], 0]; MU = U[nodeID[i], 1]; MV = U[nodeID[i], 2]; E = U[nodeID[i], 3]
        
        # Calculate kinetic energy and pressure at the node
        KE = 0.5 * ((MU)**2 + (MV)**2)/rho  # Adjust for 3D, remove MW for 2D
        pressure = (gamma - 1) * (E - KE)
        
        # Find the normals and compute ds
        if i < len(nodeID)-1:
            x1, y1 = wedge_x[i], wedge_y[i]
            x2, y2 = wedge_x[i + 1], wedge_y[i + 1]
        else:
            x1, y1 = wedge_x[i], wedge_y[i]
            x2, y2 = wedge_x[0], wedge_y[0]
        
        tx, ty = x2 - x1, y2 - y1
        length = jnp.sqrt(tx**2 + ty**2)

        nx, ny = ty/length, -tx/length # normal vector components

        ds = jnp.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Accumulate lift force
        lift += pressure * (ny) * ds  # Lift typically uses the y-component

    return lift / normaliser
@jax.jit
def calc_drag(U, x, y, nodeID):
    drag = 0
    wedge_x, wedge_y = x[nodeID], y[nodeID]
    
    # Import the conservative variables --> all are dimensional
    normaliser = 0.5 * U[0,0] * (U[0,1]/U[0,0])**2 # --> freestream
    
    # calculate lift over the nodes
    gamma = 1.4
    for i in range(len(nodeID)):
        # Get local conservative variables for each node
        rho = U[nodeID[i], 0]; MU = U[nodeID[i], 1]; MV = U[nodeID[i], 2]; E = U[nodeID[i], 3]
        
        # Calculate kinetic energy and pressure at the node
        KE = 0.5 * ((MU)**2 + (MV)**2)/rho  # Adjust for 3D, remove MW for 2D
        pressure = (gamma - 1) * (E - KE)
        
        # Find the normals and compute ds
        if i < len(nodeID)-1:
            x1, y1 = wedge_x[i], wedge_y[i]
            x2, y2 = wedge_x[i + 1], wedge_y[i + 1]
        else:
            x1, y1 = wedge_x[i], wedge_y[i]
            x2, y2 = wedge_x[0], wedge_y[0]
        
        tx, ty = x2 - x1, y2 - y1
        length = jnp.sqrt(tx**2 + ty**2)

        nx, ny = ty/length, -tx/length # normal vector components

        ds = jnp.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Accumulate lift force
        drag += pressure * (nx) * ds  # Lift typically uses the y-component

    return drag / normaliser

#---------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------- MAIN -------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------#

filename = 'flow_0.vtu'
_, data = read_vtu(filename)

array_names = ['Density', 'Velocity', 'Energy']
coord, array_data = extract_data(data, array_names)
x, y, z = coord.T
density, velocity, energy = array_data

# Conservative variable vector: U = [rho, rho*u, rho*v, rho*w, rho*E] = [rho, mom_u, mom_v, mom_w, rho*E]
mom_u, mom_v, mom_w = velocity[:,0]*density, velocity[:,1]*density, velocity[:,2]
U = jnp.stack([density, mom_u, mom_v, energy, mom_w], axis=1) # energy is alread multiplied by density

# U_df = pd.DataFrame(U, columns=['DENSITY, MOM_U, MOM_V, MOM_W, ENERGY']); U_df.to_csv('U_data.csv', index = False)

df_id = pd.read_csv('output_nodes.csv')
node_ids = df_id['NodeID'].to_numpy()

#---------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------- BEGIN OBJECTIVE FUNCTION EVALUATION AND GRADIENT CALCULATION ----------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------#
grad_start_time = time.time()
lift, del_L__del_U_autodiff = jax.value_and_grad(calc_lift, argnums=0)(U, x, y, node_ids)
grad_U_time = time.time()
print(f"J_value: {lift}")
print(del_L__del_U_autodiff)
np.savetxt(
    'del_J__del_U_autodiff.csv',
    jax.device_get(jnp.nan_to_num(del_L__del_U_autodiff, nan=0.0)),
    delimiter=','
)
print(f"Time taken for flow variable gradient calculation: {grad_U_time - grad_start_time} seconds")

# Gradients with respect to the mesh coordinates:
grad_del_L__del_x, grad_del_L__del_y = [jax.grad(calc_lift, argnums=i)(U, x, y, node_ids) for i in (1, 2)];grad_del_L__del_z = np.zeros_like(grad_del_L__del_x)

del_L__del_X_autodiff = np.column_stack([grad_del_L__del_x, grad_del_L__del_y, grad_del_L__del_z])

grad_coord_time = time.time()
print(f"Time taken for mesh gradient calculation: {grad_coord_time - grad_start_time} seconds")
np.savetxt('del_J__del_X_autodiff.csv', del_L__del_X_autodiff, delimiter=',', fmt='%.14f')

#---------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------- MAP THE GRADIENTS ONTO THE VTU FILE FOR VISUALISATION -------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------#

combined_gradients = np.hstack([
    del_L__del_U_autodiff,  # Existing flow variable gradients (e.g., density, velocity, energy)
    del_L__del_X_autodiff   # Gradients w.r.t. x, y, z (already combined)
])
field_names = [
    'Gradient_Density', 'Gradient_VelX', 'Gradient_VelY', 'Gradient_Energy',
    'Gradient_x_coord', 'Gradient_y_coord'
]
add_data_to_vtk(data, combined_gradients, field_names)
write_vtu(data, filename='wedge_with_grads.vtu')
print(f'VTU file with gradients saved as wedge_with_grads.vtu')
