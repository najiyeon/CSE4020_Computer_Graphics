# CSE4020_Computer_Graphics
*Hanyang University, Computer Graphics, Spring 2023, by prof. Yoonsang Lee*

<br/>

## [Project 01 : Basic OpenGL viewer](https://github.com/najiyeon/CSE4020_Computer_Graphics/blob/main/project1/main.py)
1. A **rectangular grid** with lines (not polygons) on _**xz plane**_
   -	Implemented with prepare_vao_grid(), draw_grid(), and draw_grid_array()

<br/>

2. **Orbit** : Rotate the camera around the target point by changing azimuth / elevation angles when the _**mouse left button**_ is clicked and dragged   

<img src="https://github.com/najiyeon/CSE4020_Computer_Graphics/assets/113894257/717523a7-75ef-4786-9cb0-582f9d084cb0" width="800"></img>

<br/>

3. **Pan** : Move both the target point and camera in left, right, up and down direction of the camera when the _**mouse right button**_ is clicked and dragged   

<img src="https://github.com/najiyeon/CSE4020_Computer_Graphics/assets/113894257/5d15b36c-2d24-4b44-b869-a149e0d32cad" width="800"></img>

<br/>

4. **Zoom** : Move the camera forward toward the target point (zoom in) and backward away from the target point (zoom out) when the _**mouse wheel**_ is rotated   

<img src="https://github.com/najiyeon/CSE4020_Computer_Graphics/assets/113894257/a961ebaf-2592-4d31-84f6-1a51967f031d" width="800"></img>

<br/>

5. Toggle **perspective projection / orthogonal projection** by pressing _**‘v’ key**_   

<img src="https://github.com/najiyeon/CSE4020_Computer_Graphics/assets/113894257/e2e4dacd-0be8-4503-b27f-7920d1b6d69f" width="800"></img>

<br/>

## [Project 02 : Obj viewer & drawing a hierarchical model](https://github.com/najiyeon/CSE4020_Computer_Graphics/blob/main/project2/main.py)
1. **Single mesh rendering mode** : When an obj file is _**dropped down**_, rendering is performed with vertex position, vertex normal, and face information, and obj file information is printed.   

<img src="https://github.com/najiyeon/CSE4020_Computer_Graphics/assets/113894257/8a18a80e-018e-4701-bdcb-0182bafe78e5" width="800"></img>

<br>

2. **Animating hierarchical model rendering mode** : When a user presses a _**key ‘h’**_ on your viewer, the program runs in “animating hierarchical model rendering mode”.   

<img src="https://github.com/najiyeon/CSE4020_Computer_Graphics/assets/113894257/a7b51bbe-d5b6-4a08-bb2b-1fbf6083d021" width="800"></img>

<br>

**vidoe capture of the animating hierarchical model**   

https://github.com/najiyeon/CSE4020_Computer_Graphics/assets/113894257/157a2d93-3c7b-4e59-bdba-611d8b3ed7aa

<br>

3. **Hierarchical model** : The model has a hierarchy of 3 levels, and each node has 2 child nodes. And all the child body parts move relative to the parent body parts.   

<img src="https://github.com/najiyeon/CSE4020_Computer_Graphics/assets/113894257/67d24246-a958-4316-8182-b9623a7ecf8f" width="500"></img>

<br>
<!--
## Project 03 : Bvh Viewer
-->
<br>
