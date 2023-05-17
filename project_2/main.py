from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os

g_cam_azimuth = -45.0
g_cam_elevation = 45.0
g_cam_distance = 5.0
g_target_pos = glm.vec3(0.0, 0.0, 0.0)
g_cam_pos = glm.vec3(g_cam_distance*np.sin(np.radians(g_cam_azimuth))*np.cos(np.radians(g_cam_elevation)),
                        g_cam_distance*np.sin(np.radians(g_cam_elevation)),
                        g_cam_distance*np.cos(np.radians(g_cam_azimuth))*np.cos(np.radians(g_cam_elevation)))
g_cam_pos += g_target_pos
g_up_vector = glm.vec3(0.0, 1.0, 0.0)
g_press_left = 0
g_press_right = 0
g_last_x = 240.0
g_last_y = 240.0
g_x_offset = 0.0
g_y_offset = 0.0

# # initialize model matrix
# M = glm.mat4()

# # view matrix
# V = glm.lookAt(g_cam_pos, g_target_pos, g_up_vector)

# initialize projection matrix
ortho_height = 10.
ortho_width = ortho_height * 800/800    # initial width/height

# # now projection matrix P is a global variable so that it can be accessed from main() and key_callback()
# g_P = glm.mat4()
# g_P = glm.perspective(45, 1, 1, 800)

# key callback 'v'
g_key_v = 0

############################ project 2 ############################
# obj file name
g_obj_file_name = ''

# obj vertex position
g_vertex_pos = []

# obj vertex normal
g_vertex_normal = []

# obj face index
g_face_index = []

# obj vertex array
g_vertex_array = glm.array(glm.float32)

# obj face vertex count
g_face_vertex_count = [0, 0, 0]

# draw count
g_count = [0, 0, 0, 0, 0, 0, 0]
temp_count = 0

# key callback 'z'
g_key_z = 0
# key callback 'h'
g_key_h = 0
# drop callback check
g_drop = 0

# shader for project 1
g_vertex_shader_src_grid = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src_grid = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

# shader for project 2
g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos;
layout (location = 1) in vec3 vin_normal;

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 color;

void main()
{
    // light and material properties
    vec3 light_pos = vec3(3,2,4);
    vec3 light_color = vec3(1,1,1);
    vec3 light_pos2 = vec3(3,2,-4);
    vec3 light_color2 = vec3(1,1,1);

    vec3 material_color = color;
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;
    vec3 light_ambient2 = 0.1*light_color2;
    vec3 light_diffuse2 = light_color2;
    vec3 light_specular2 = light_color2;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // for non-metal material
    vec3 material_specular2 = light_color2;

    // ambient
    vec3 ambient = light_ambient * material_ambient;
    vec3 ambient2 = light_ambient2 * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);
    vec3 light_dir2 = normalize(light_pos2 - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;
    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse2 = diff2 * light_diffuse2 * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;
    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    vec3 specular2 = spec2 * light_specular2 * material_specular2;

    vec3 fcolor = ambient + diffuse + specular + ambient2 + diffuse2 + specular2;
    FragColor = vec4(fcolor, 1.);
}
'''

class Node:
    def __init__(self, parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color

def draw_node(vao, node, VP, MVP_loc, M_loc, view_pos_loc, color_loc):
    global temp_count, g_cam_pos
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    M = node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glUniform3f(view_pos_loc, g_cam_pos.x, g_cam_pos.y, g_cam_pos.z)

    glDrawArrays(GL_TRIANGLES, 0, temp_count)

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def parse(file):
    global g_vertex_pos, g_vertex_normal, g_face_index, g_face_vertex_count, g_vertex_array

    objData = file.read()
    lines = objData.splitlines()

    g_vertex_pos = []
    g_vertex_normal = []
    g_face_index = []
    g_face_vertex_count = [0, 0, 0]
    g_vertex_array = []

    for line in lines:
        tokens = line.split()

        if len(tokens) == 0:
            continue

        if tokens[0] == 'v':
            g_vertex_pos.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        elif tokens[0] == 'vn':
            g_vertex_normal.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
        elif tokens[0] == 'f':
            if len(tokens) == 4: # 3 vertex
                g_face_vertex_count[0] += 1
                for face_token in tokens[1:]:
                    face_data = face_token.split('//')
                    g_face_index.append([int(face_data[0]), int(face_data[1])])
            elif len(tokens) == 5: # 4 vertex
                g_face_vertex_count[1] += 1
                for face_token in tokens[1], tokens[2], tokens[3]:
                    face_data = face_token.split('//')
                    g_face_index.append([int(face_data[0]), int(face_data[1])])
                for face_token in tokens[1], tokens[3], tokens[4]:
                    face_data = face_token.split('//')
                    g_face_index.append([int(face_data[0]), int(face_data[1])])
            else: # more than 4 vertex
                g_face_vertex_count[2] += 1
                for i in range(2, len(tokens) - 1):
                    for face_token in tokens[1], tokens[i], tokens[i+1]:
                        face_data = face_token.split('//')
                        g_face_index.append([int(face_data[0]), int(face_data[1])])

    vertex_array = []
    i = 0;
    while i < len(g_face_index):
        vertex_array = vertex_array + g_vertex_pos[g_face_index[i][0] - 1]
        vertex_array = vertex_array + g_vertex_normal[g_face_index[i][1] - 1]
        i = i + 1

    # convert python list to numpy array
    vertex_array_np = np.array(vertex_array, dtype=glm.float32)

    # convert numpy array to glm array
    g_vertex_array = glm.array(vertex_array_np)

    return g_vertex_array

def key_callback(window, key, scancode, action, mods):
    global g_key_v, g_key_z, g_drop, g_key_h
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    else:
        if action==GLFW_RELEASE and key==GLFW_KEY_V:
            g_key_v = 1 - g_key_v
            # if g_P == glm.ortho(-ortho_width*.5,ortho_width*.5, -ortho_height*.5,ortho_height*.5, -400,400):
            #     g_P = glm.perspective(45, 1, 1, 800)
            # else:
            #     g_P = glm.ortho(-ortho_width*.5,ortho_width*.5, -ortho_height*.5,ortho_height*.5, -400,400)
    
        elif action==GLFW_RELEASE and key==GLFW_KEY_Z:
            g_key_z = 1 - g_key_z
        elif action==GLFW_RELEASE and key==GLFW_KEY_H:
            g_key_h = 1 - g_key_h
            g_drop = 0
        
def mouse_button_callback(window, button, action, mods):
    #called whenever a mouse button is pressed or released
    global g_press_left, g_press_right

    #checks which button was pressed/released (left or right) 
    if button==GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            g_press_left = 1
        elif action==GLFW_RELEASE:
            g_press_left = 0
    
    if button==GLFW_MOUSE_BUTTON_RIGHT:
        if action==GLFW_PRESS:
            g_press_right = 1
        elif action==GLFW_RELEASE:
            g_press_right = 0

def cursor_position_callback(window, xpos, ypos):
    # called whenever the mouse cursor moves
    global V, g_press_left, g_press_right, g_last_x, g_last_y, g_x_offset, g_y_offset, g_cam_azimuth, g_cam_elevation, g_cam_distance, g_cam_pos, g_target_pos, g_up_vector

    g_x_offset = xpos - g_last_x
    g_y_offset = g_last_y - ypos
    g_last_x = xpos
    g_last_y = ypos

    # Orbit
    if g_press_left == 1:
        sensitivity = 0.1

        g_cam_azimuth -= g_x_offset * sensitivity
        g_cam_elevation -= g_y_offset * sensitivity

        # Make sure that when elevation is out of bounds, screen doesn't get flipped
        if g_cam_elevation > 89.0:
            g_cam_elevation = 89.0
        if g_cam_elevation < -89.0:
            g_cam_elevation = -89.0

        g_cam_pos = glm.vec3(g_cam_distance*np.sin(np.radians(g_cam_azimuth))*np.cos(np.radians(g_cam_elevation)),
                        g_cam_distance*np.sin(np.radians(g_cam_elevation)),
                        g_cam_distance*np.cos(np.radians(g_cam_azimuth))*np.cos(np.radians(g_cam_elevation)))
        g_cam_pos += g_target_pos

        # V = glm.lookAt(g_cam_pos, g_target_pos, g_up_vector)

    # Pan
    elif g_press_right == 1:
        sensitivity = 0.01

        g_cam_dir = glm.normalize(g_cam_pos - g_target_pos)
        right = glm.normalize(glm.cross(g_up_vector, g_cam_dir))
        up = glm.normalize(glm.cross(g_cam_dir, right))

        T = glm.translate(-g_x_offset * sensitivity * right + -g_y_offset * sensitivity * up)

        g_cam_pos = T * g_cam_pos
        g_target_pos = T * g_target_pos

        # V = glm.lookAt(g_cam_pos, g_target_pos, g_up_vector)

def scroll_callback(window, xoffset, yoffset):
    global V, g_cam_distance, g_cam_pos, g_target_pos, g_up_vector, g_cam_azimuth, g_cam_elevation

    sensitivity = 0.3

    # Zoom
    if yoffset < 0: # scroll up (zoom out)
        g_cam_distance += -yoffset * sensitivity

    else: # scroll down (zoom in)
        if g_cam_distance > 1.0:
            g_cam_distance -= yoffset * sensitivity
        else:
            g_cam_distance = g_cam_distance

    g_cam_pos = glm.vec3(g_cam_distance*np.sin(np.radians(g_cam_azimuth))*np.cos(np.radians(g_cam_elevation)),
                    g_cam_distance*np.sin(np.radians(g_cam_elevation)),
                    g_cam_distance*np.cos(np.radians(g_cam_azimuth))*np.cos(np.radians(g_cam_elevation)))
    g_cam_pos += g_target_pos
    # V = glm.lookAt(g_cam_pos, g_target_pos, glm.vec3(0,1,0))

def drop_callback(window, paths):
    # single mash rendering mode
    global g_obj_file_name, g_drop, g_key_z

    g_drop = 1
    g_key_z = 0
    objFile = open(paths[0], 'r')

    g_obj_file_name = os.path.basename(paths[0])
    parse(objFile)

    objFile.close()

    print_information()

def prepare_vao_obj():
    global g_vertex_array

    # create and activate VAO & VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex & index data to VBO
    glBufferData(GL_ARRAY_BUFFER, g_vertex_array.nbytes, g_vertex_array.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_obj(vao, MVP, MVP_loc, M, M_loc, view_pos, view_pos_loc):
    global g_face_index

    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    glUniform3fv(view_pos_loc, 1, glm.value_ptr(view_pos))
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, len(g_face_index))

def print_information():
    global g_obj_file_name, g_face_index
    
    print("Obj file name: " + g_obj_file_name)
    print("Total number of faces: " + str(g_face_vertex_count[0] + g_face_vertex_count[1] + g_face_vertex_count[2]))
    print("Number of faces with 3 vertices: " + str(g_face_vertex_count[0]))
    print("Number of faces with 4 vertices: " + str(g_face_vertex_count[1]))
    print("Number of faces more than 4 vertices: " + str(g_face_vertex_count[2]))

def prepare_vao_road():
    global g_vertex_array, g_face_index

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'obj_file', 'road.obj')

    objFile = open(file_path, 'r')
    vertex_array = parse(objFile)
    objFile.close()

    g_count[0] = len(g_face_index)

    # create and activate VAO & VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex & index data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_helicopter():
    global g_vertex_array, g_face_index

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'obj_file', 'helicopter.obj')

    objFile = open(file_path, 'r')
    vertex_array = parse(objFile)
    objFile.close()

    g_count[1] = len(g_face_index)

    # create and activate VAO & VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex & index data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_wing():
    global g_vertex_array, g_face_index

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'obj_file', 'helicopter_wing.obj')

    objFile = open(file_path, 'r')
    vertex_array = parse(objFile)
    objFile.close()

    vertex_array = g_vertex_array
    g_count[2] = len(g_face_index)

    # create and activate VAO & VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex & index data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_eagle():
    global g_vertex_array, g_face_index

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'obj_file', 'eagle.obj')

    objFile = open(file_path, 'r')
    vertex_array = parse(objFile)
    objFile.close()

    vertex_array = g_vertex_array
    g_count[3] = len(g_face_index)

    # create and activate VAO & VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex & index data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_tank():
    global g_vertex_array, g_face_index

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'obj_file', 'tank.obj')

    objFile = open(file_path, 'r')
    vertex_array = parse(objFile)
    objFile.close()

    vertex_array = g_vertex_array
    g_count[4] = len(g_face_index)

    # create and activate VAO & VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex & index data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_top():
    global g_vertex_array, g_face_index

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'obj_file', 'tank_top.obj')

    objFile = open(file_path, 'r')
    vertex_array = parse(objFile)
    objFile.close()

    vertex_array = g_vertex_array
    g_count[5] = len(g_face_index)

    # create and activate VAO & VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex & index data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_drone():
    global g_vertex_array, g_face_index

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'obj_file', 'drone.obj')

    objFile = open(file_path, 'r')
    vertex_array = parse(objFile)
    objFile.close()

    vertex_array = g_vertex_array
    g_count[6] = len(g_face_index)

    # create and activate VAO & VBO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex & index data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3 * glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -50.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         50.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 50.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, -50.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 50.0,  0.0, 0.0, 1.0, # z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
        -10.0, 0.0, 0.0,  0.5, 0.5, 0.5, # x-axis start
        10.0, 0.0, 0.0,  0.5, 0.5, 0.5, # x-axis end
        0.0, 0.0, -10.0,  0.5, 0.5, 0.5, # z-axis start
        0.0, 0.0, 10.0,  0.5, 0.5, 0.5, # z-axis end
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_frame(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 6)

def draw_grid(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 4)

def draw_grid_array(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    for i in range(-50, 50):
        for j in range(-50, 50):
            if i != 0 and j != 0:
                MVP_grid = MVP * glm.translate(glm.vec3(0.5*i, 0, 0.5*j))
                glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP_grid))
                glDrawArrays(GL_LINES, 0, 4)

def main():
    global g_drop, temp_count, g_count, M, g_cam_pos, g_cam_azimuth
    
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, 'Project 2', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback);

    #register mouse callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback)

    # register cursor position callbacks
    glfwSetCursorPosCallback(window, cursor_position_callback)

    # register scroll callbacks
    glfwSetScrollCallback(window, scroll_callback)

    # register drop callbacks
    glfwSetDropCallback(window, drop_callback)

    # load shaders
    shader_program_for_grid = load_shaders(g_vertex_shader_src_grid, g_fragment_shader_src_grid)
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc_for_grid = glGetUniformLocation(shader_program_for_grid, 'MVP')
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    M_loc = glGetUniformLocation(shader_program, 'M')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')
    color_loc = glGetUniformLocation(shader_program, 'color')
    
    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_grid = prepare_vao_grid()

    vao_road = prepare_vao_road()
    vao_helicopter = prepare_vao_helicopter()
    vao_wing = prepare_vao_wing()
    vao_eagle = prepare_vao_eagle()
    vao_tank = prepare_vao_tank()
    vao_top = prepare_vao_top()
    vao_drone = prepare_vao_drone()

    # create a hierarchical model
    road = Node(None, glm.mat4(), glm.vec3(.6,.6,.6))
    helicopter = Node(road, glm.mat4(), glm.vec3(.5,.5,.5))
    wing = Node(helicopter, glm.mat4(), glm.vec3(.9,.9,.9))
    eagle = Node(helicopter, glm.scale((.1,.1,.1)) * glm.rotate(glm.radians(90), glm.vec3(0,1,0)), glm.vec3(1,1,1))
    tank = Node(road, glm.mat4(), glm.vec3(.5,.5,.5))
    top = Node(tank, glm.mat4(), glm.vec3(.9,.9,.9))
    drone = Node(tank, glm.scale((.5,.5,.5)), glm.vec3(.8,.8,.8))

    # # viewport
    # glViewport(100,100, 200,200)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode" if g_key_z is 1
        if g_key_z == 1:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else: # render in solid mode
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glUseProgram(shader_program_for_grid)

        # set projection matrix
        if g_key_v == 0:
            P = glm.perspective(45, 1, 1, 800)
        else:
            P = glm.ortho(-ortho_width*.5,ortho_width*.5, -ortho_height*.5,ortho_height*.5, -400,400)

        # set view matrix
        V = glm.lookAt(g_cam_pos, g_target_pos, g_up_vector)

        # set model matrix
        M = glm.translate(glm.vec3(0,0,0))

        # draw frame
        draw_frame(vao_frame, P*V*M, MVP_loc_for_grid)

        # draw grid
        draw_grid_array(vao_grid, P*V*M, MVP_loc_for_grid)

        glUseProgram(shader_program)

        view_pos = glm.vec3(g_cam_pos.x, g_cam_pos.y, g_cam_pos.z)

        # draw object
        if g_drop == 1:
            glUniform3f(color_loc, 1.0, 1.0, 1.0)
            vao_obj = prepare_vao_obj()
            draw_obj(vao_obj, P*V*M, MVP_loc, M, M_loc, view_pos, view_pos_loc)

        elif g_key_h == 1:
            t = glfwGetTime()

            # set local transformations of each node
            # root node (road) just stays at the origin
            # helicopter rotates above the road
            helicopter.set_transform(glm.rotate(t, (0,-1,0)) * glm.translate((5,8,0)))
            # wing rotates about the helicopter
            wing.set_transform(glm.rotate(t*10, (0,1,0)) * glm.translate((0,0,0)))
            # eagle follows the helicopter
            eagle.set_transform(glm.translate((0,1,-5)) * glm.rotate(np.sin(t), (1,0,0)))
            # tank moves forward and backward on the road
            tank.set_transform(glm.translate((-5,0.8,5*np.sin(t))))
            # top rotates about the tank
            top.set_transform(glm.translate((0.5,0,0)) * glm.rotate(t, (0,-1,0)))
            # drone rotates about the tank
            drone.set_transform(glm.rotate(t*2, (0,1,0)) * glm.translate((0,4,-1)))

            road.update_tree_global_transform()

            # draw the hierarchical model
            temp_count = g_count[0]
            draw_node(vao_road, road, P*V, MVP_loc, M_loc, view_pos_loc, color_loc)
            temp_count = g_count[1]
            draw_node(vao_helicopter, helicopter, P*V, MVP_loc, M_loc, view_pos_loc, color_loc)
            temp_count = g_count[2]
            draw_node(vao_wing, wing, P*V, MVP_loc, M_loc, view_pos_loc, color_loc)
            temp_count = g_count[3]
            draw_node(vao_eagle, eagle, P*V, MVP_loc, M_loc, view_pos_loc, color_loc)
            temp_count = g_count[4]
            draw_node(vao_tank, tank, P*V, MVP_loc, M_loc, view_pos_loc, color_loc)
            temp_count = g_count[5]
            draw_node(vao_top, top, P*V, MVP_loc, M_loc, view_pos_loc, color_loc)
            temp_count = g_count[6]
            draw_node(vao_drone, drone, P*V, MVP_loc, M_loc, view_pos_loc, color_loc)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()

