from OpenGL.GL import *
from glfw.GLFW import *
import glm
import numpy as np

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

# view matrix
V = glm.lookAt(g_cam_pos, g_target_pos, g_up_vector)

# initialize projection matrix
ortho_height = 10.
ortho_width = ortho_height * 800/800    # initial width/height

# now projection matrix P is a global variable so that it can be accessed from main() and key_callback()
g_P = glm.mat4()
g_P = glm.perspective(45, 1, 1, 800)

g_vertex_shader_src = '''
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

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

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

def key_callback(window, key, scancode, action, mods):
    global g_cam_ang, g_cam_height, g_P, ortho_width, ortho_height
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    else:
        if action==GLFW_RELEASE and key==GLFW_KEY_V:
            if g_P == glm.ortho(-ortho_width*.5,ortho_width*.5, -ortho_height*.5,ortho_height*.5, -400,400):
                g_P = glm.perspective(45, 1, 1, 800)
            else:
                g_P = glm.ortho(-ortho_width*.5,ortho_width*.5, -ortho_height*.5,ortho_height*.5, -400,400)

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

        V = glm.lookAt(g_cam_pos, g_target_pos, g_up_vector)

    # Pan
    elif g_press_right == 1:
        sensitivity = 0.01

        g_cam_dir = glm.normalize(g_cam_pos - g_target_pos)
        right = glm.normalize(glm.cross(g_up_vector, g_cam_dir))
        up = glm.normalize(glm.cross(g_cam_dir, right))

        T = glm.translate(-g_x_offset * sensitivity * right + -g_y_offset * sensitivity * up)

        g_cam_pos = T * g_cam_pos
        g_target_pos = T * g_target_pos

        V = glm.lookAt(g_cam_pos, g_target_pos, g_up_vector)

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
    V = glm.lookAt(g_cam_pos, g_target_pos, glm.vec3(0,1,0))

def prepare_vao_cube():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
        # position            color
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
         0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v2
         0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v1
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
        -0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v3
         0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v2
                    
        -0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v4
         0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v5
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
                    
        -0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v4
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
        -0.5 , -0.5 , -0.5 ,  1, 1, 1, # v7
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
         0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v1
         0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v5
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
         0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v5
        -0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v4
 
        -0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v3
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
         0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v2
                    
        -0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v3
        -0.5 , -0.5 , -0.5 ,  1, 1, 1, # v7
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
                    
         0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v1
         0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v2
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
                    
         0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v1
         0.5 , -0.5 , -0.5 ,  1, 1, 1, # v6
         0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v5
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
        -0.5 , -0.5 , -0.5 ,  1, 1, 1, # v7
        -0.5 , -0.5 ,  0.5 ,  1, 1, 1, # v3
                    
        -0.5 ,  0.5 ,  0.5 ,  1, 1, 1, # v0
        -0.5 ,  0.5 , -0.5 ,  1, 1, 1, # v4
        -0.5 , -0.5 , -0.5 ,  1, 1, 1, # v7
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

def draw_cube(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_TRIANGLES, 0, 36)

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
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, 'Project 1', None, None)
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

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_cube = prepare_vao_cube()
    vao_frame = prepare_vao_frame()
    vao_grid = prepare_vao_grid()

    # # viewport
    # glViewport(100,100, 200,200)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode"
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        glUseProgram(shader_program)

        M = glm.mat4()

        # draw cube array w.r.t. the current frame MVP
        draw_cube(vao_cube, g_P*V*M, MVP_loc)

        # draw world frame
        draw_frame(vao_frame, g_P*V*M, MVP_loc)

        # draw grid
        draw_grid_array(vao_grid, g_P*V*M, MVP_loc)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()

