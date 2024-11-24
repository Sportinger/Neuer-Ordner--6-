import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random

# Parameters
layer_count = 9  # Default number of layers (should be an odd number for symmetry)
outline_thickness = 2  # Thickness of the white outlines (will be adjusted based on window size)

# Camera parameters
camera_distance = layer_count * 4  # Initial distance from the center
camera_angle_x = 0.0  # Rotation around the x-axis (up/down)
camera_angle_y = 0.0  # Rotation around the y-axis (left/right)
fov = 45.0  # Field of view angle

# Window size
viewport = (800, 600)  # Initial window size

# GUI elements
button_rect = pygame.Rect(10, 10, 120, 30)  # Position and size of the randomize button
random_slider_rect = pygame.Rect(10, 50, 200, 20)  # Position and size of the randomization slider
layer_slider_rect = pygame.Rect(10, 80, 200, 20)    # Position and size of the layers slider

# Randomization intensity (0 to 1)
random_intensity = 0.5  # Default intensity

# Layers slider value
layer_slider_value = 9  # Default layer count

# Initialize Pygame and OpenGL
def init():
    global font  # Declare font as global
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode(viewport, DOUBLEBUF | OPENGL | RESIZABLE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glLineWidth(outline_thickness)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Initialize font
    font = pygame.font.SysFont('Arial', 16)

# Generate the diamond-shaped structure
def generate_diamond_structure(layer_count):
    cubes = []
    max_layer = layer_count // 2
    for layer in range(layer_count):
        if layer <= max_layer:
            # Expanding layers
            size = 2 * layer + 1
            offset = -layer
            y = layer
        else:
            # Contracting layers
            layer_from_top = layer_count - 1 - layer
            size = 2 * layer_from_top + 1
            offset = -layer_from_top
            y = layer
        for x in range(size):
            for z in range(size):
                if x == 0 or x == size - 1 or z == 0 or z == size - 1:
                    cubes.append((x + offset, y, z + offset))
    return cubes

# Generate a random pattern structure with adjustable intensity
def generate_random_structure(layer_count, intensity):
    cubes = []
    size = layer_count  # Use layer_count as the size of the structure
    probability = intensity  # Probability of placing a cube

    for y in range(size):
        for x in range(-size // 2, size // 2 + 1):
            for z in range(-size // 2, size // 2 + 1):
                if random.random() < probability:
                    cubes.append((x, y, z))
    # Remove non-visible cubes
    cubes = remove_interior_cubes(cubes)
    return cubes

# Remove interior cubes that are not visible from any angle
def remove_interior_cubes(cubes):
    cube_set = set(cubes)
    visible_cubes = set()
    directions = [(-1, 0, 0), (1, 0, 0),
                  (0, -1, 0), (0, 1, 0),
                  (0, 0, -1), (0, 0, 1)]
    for cube in cubes:
        x, y, z = cube
        is_exposed = False
        for dx, dy, dz in directions:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor not in cube_set:
                is_exposed = True
                break
        if is_exposed:
            visible_cubes.add(cube)
    return list(visible_cubes)

# Remove duplicate edges
def remove_duplicate_edges(edges):
    edge_dict = {}
    for edge in edges:
        # Edge represented as sorted tuple of vertices
        v1, v2 = edge
        key = tuple(sorted((tuple(v1), tuple(v2))))
        edge_dict[key] = v1, v2  # Keep one instance
    unique_edges = list(edge_dict.values())
    return unique_edges

# Prepare buffers for GPU rendering
def create_vbo(cubes):
    vertices = []
    edges = []
    for cube in cubes:
        x, y, z = cube
        # Define the 8 vertices of the cube
        cube_vertices = [
            (x + 0.5, y - 0.5, z - 0.5),
            (x + 0.5, y + 0.5, z - 0.5),
            (x - 0.5, y + 0.5, z - 0.5),
            (x - 0.5, y - 0.5, z - 0.5),
            (x + 0.5, y - 0.5, z + 0.5),
            (x + 0.5, y + 0.5, z + 0.5),
            (x - 0.5, y - 0.5, z + 0.5),
            (x - 0.5, y + 0.5, z + 0.5)
        ]
        # Surfaces of the cube
        surfaces = (
            (0, 1, 2, 3),
            (3, 2, 7, 6),
            (6, 7, 5, 4),
            (4, 5, 1, 0),
            (1, 5, 7, 2),
            (4, 0, 3, 6)
        )
        # Edges of the cube
        cube_edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 7),
            (3, 6)
        )

        # Add faces
        for surface in surfaces:
            for vertex in surface:
                vertices.extend(cube_vertices[vertex])

        # Collect edges
        for edge in cube_edges:
            edges.append((cube_vertices[edge[0]], cube_vertices[edge[1]]))

    # Remove duplicate edges
    unique_edges = remove_duplicate_edges(edges)

    # Prepare data for VBOs
    vertices = np.array(vertices, dtype=np.float32)
    edge_vertices = []
    for v1, v2 in unique_edges:
        edge_vertices.extend(v1)
        edge_vertices.extend(v2)
    edge_vertices = np.array(edge_vertices, dtype=np.float32)

    # Create VBOs
    vertex_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    edge_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, edge_vbo)
    glBufferData(GL_ARRAY_BUFFER, edge_vertices.nbytes, edge_vertices, GL_STATIC_DRAW)

    return vertex_vbo, len(vertices), edge_vbo, len(edge_vertices)

# Draw the cubes using VBOs
def draw_vbo(vertex_vbo, vertex_count, edge_vbo, edge_count):
    # Draw cube surfaces
    glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glColor3f(0.0, 0.0, 0.0)  # Black cubes
    glDrawArrays(GL_QUADS, 0, vertex_count // 3)
    glDisableClientState(GL_VERTEX_ARRAY)

    # Draw edges
    glBindBuffer(GL_ARRAY_BUFFER, edge_vbo)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glColor3f(1.0, 1.0, 1.0)  # White outlines
    glDrawArrays(GL_LINES, 0, edge_count // 3)
    glDisableClientState(GL_VERTEX_ARRAY)

    glBindBuffer(GL_ARRAY_BUFFER, 0)

# Draw the on-screen GUI elements using OpenGL
def draw_gui():
    # Switch to orthographic projection
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, viewport[0], viewport[1], 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Disable depth test to draw GUI elements over the 3D scene
    glDisable(GL_DEPTH_TEST)

    # Draw Randomize Button
    draw_rect(button_rect, (0.8, 0.8, 0.8))  # Gray background
    draw_rect_outline(button_rect, (0.0, 0.0, 0.0))  # Black border
    draw_text('Randomize', button_rect.centerx, button_rect.centery, (0, 0, 0))

    # Draw Randomization Slider
    draw_rect(random_slider_rect, (0.8, 0.8, 0.8))  # Gray background
    draw_rect_outline(random_slider_rect, (0.0, 0.0, 0.0))  # Black border
    # Slider handle position
    handle_x = random_slider_rect.x + int(random_intensity * random_slider_rect.width)
    handle_rect = pygame.Rect(handle_x - 5, random_slider_rect.y - 5, 10, random_slider_rect.height + 10)
    draw_rect(handle_rect, (0.6, 0.6, 0.6))  # Darker gray handle
    draw_rect_outline(handle_rect, (0.0, 0.0, 0.0))  # Black border
    draw_text('Randomness', random_slider_rect.left, random_slider_rect.top - 10, (0, 0, 0), align='left')

    # Draw Layers Slider
    draw_rect(layer_slider_rect, (0.8, 0.8, 0.8))  # Gray background
    draw_rect_outline(layer_slider_rect, (0.0, 0.0, 0.0))  # Black border
    # Slider handle position
    layer_slider_normalized = (layer_slider_value - 3) / (21 - 3)  # Assuming layer_count between 3 and 21
    handle_x_layer = layer_slider_rect.x + int(layer_slider_normalized * layer_slider_rect.width)
    handle_rect_layer = pygame.Rect(handle_x_layer - 5, layer_slider_rect.y - 5, 10, layer_slider_rect.height + 10)
    draw_rect(handle_rect_layer, (0.6, 0.6, 0.6))  # Darker gray handle
    draw_rect_outline(handle_rect_layer, (0.0, 0.0, 0.0))  # Black border
    draw_text('Layers', layer_slider_rect.left, layer_slider_rect.top - 10, (0, 0, 0), align='left')

    # Restore previous matrices
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()

# Helper function to draw rectangles using OpenGL
def draw_rect(rect, color):
    glColor4f(*color, 1.0)
    glBegin(GL_QUADS)
    glVertex2f(rect.left, rect.top)
    glVertex2f(rect.right, rect.top)
    glVertex2f(rect.right, rect.bottom)
    glVertex2f(rect.left, rect.bottom)
    glEnd()

# Helper function to draw rectangle outlines using OpenGL
def draw_rect_outline(rect, color):
    glColor4f(*color, 1.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(rect.left, rect.top)
    glVertex2f(rect.right, rect.top)
    glVertex2f(rect.right, rect.bottom)
    glVertex2f(rect.left, rect.bottom)
    glEnd()

# Draw text using Pygame font and OpenGL texture
def draw_text(message, x, y, color=(0, 0, 0), align='center'):
    # Render text using Pygame font
    text_surface = font.render(message, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    text_width, text_height = text_surface.get_size()

    # Generate a texture ID
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, text_width, text_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    # Calculate position based on alignment
    if align == 'center':
        pos_x = x - text_width // 2
    elif align == 'left':
        pos_x = x
    elif align == 'right':
        pos_x = x - text_width
    pos_y = y - text_height // 2

    # Enable blending
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Enable textures
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Draw textured quad
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(pos_x, pos_y)
    glTexCoord2f(1, 1)
    glVertex2f(pos_x + text_width, pos_y)
    glTexCoord2f(1, 0)
    glVertex2f(pos_x + text_width, pos_y + text_height)
    glTexCoord2f(0, 0)
    glVertex2f(pos_x, pos_y + text_height)
    glEnd()

    # Disable textures and blending
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)

    # Delete the texture (pass as a sequence)
    glDeleteTextures([texture_id])  # Corrected line

# Camera and controls
def handle_mouse_movement(delta_x, delta_y):
    global camera_angle_x, camera_angle_y
    camera_angle_y += delta_x * 0.2  # Adjust sensitivity as needed
    camera_angle_x += delta_y * 0.2
    camera_angle_x = max(-90, min(90, camera_angle_x))  # Limit to prevent flipping

def handle_zoom(delta, ctrl_pressed):
    global camera_distance, fov
    if ctrl_pressed:
        # Adjust focal length (field of view)
        fov += delta * 2  # Adjust sensitivity as needed
        fov = max(10, min(100, fov))  # Clamp FOV to reasonable values
    else:
        # Adjust camera distance
        camera_distance += delta * 2  # Adjust sensitivity as needed
        camera_distance = max(2, camera_distance)  # Prevent camera from getting too close

def resize_viewport(width, height):
    global viewport, button_rect, random_slider_rect, layer_slider_rect, outline_thickness
    viewport = (width, height)
    glViewport(0, 0, width, height)
    # Adjust GUI element positions if needed
    button_rect = pygame.Rect(10, 10, 120, 30)
    random_slider_rect = pygame.Rect(10, 50, 200, 20)
    layer_slider_rect = pygame.Rect(10, 80, 200, 20)
    # Adjust outline_thickness based on window size (e.g., proportional to height)
    outline_thickness = max(1, int(height / 300))  # Adjust the divisor to change sensitivity
    glLineWidth(outline_thickness)

def main():
    global layer_count, camera_distance, random_intensity, layer_slider_value
    init()
    cubes = generate_diamond_structure(layer_count)
    vertex_vbo, vertex_count, edge_vbo, edge_count = create_vbo(cubes)
    clock = pygame.time.Clock()
    mouse_down = False
    last_pos = (0, 0)
    randomize = False
    slider_active = False
    layer_slider_active = False

    while True:
        ctrl_pressed = pygame.key.get_mods() & KMOD_CTRL
        for event in pygame.event.get():
            if event.type == QUIT:
                glDeleteBuffers(1, [vertex_vbo])
                glDeleteBuffers(1, [edge_vbo])
                pygame.quit()
                return
            elif event.type == VIDEORESIZE:
                resize_viewport(event.w, event.h)
                pygame.display.set_mode((event.w, event.h), DOUBLEBUF | OPENGL | RESIZABLE)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    if button_rect.collidepoint(event.pos):
                        randomize = True
                    elif random_slider_rect.collidepoint(event.pos):
                        slider_active = True
                        # Update random_intensity based on mouse position
                        mouse_x = event.pos[0]
                        slider_value = (mouse_x - random_slider_rect.x) / random_slider_rect.width
                        random_intensity = max(0.0, min(1.0, slider_value))
                    elif layer_slider_rect.collidepoint(event.pos):
                        layer_slider_active = True
                        # Update layer_slider_value based on mouse position
                        mouse_x = event.pos[0]
                        slider_value = (mouse_x - layer_slider_rect.x) / layer_slider_rect.width
                        # Assuming layer_count ranges from 3 to 21 and must be odd
                        raw_value = 3 + int(slider_value * (21 - 3))
                        # Make sure it's odd and within range
                        raw_value = max(3, min(21, raw_value))
                        if raw_value % 2 == 0:
                            raw_value += 1 if raw_value < 21 else -1
                        layer_slider_value = raw_value
                    else:
                        mouse_down = True
                        last_pos = event.pos
                elif event.button == 4:
                    handle_zoom(-1, ctrl_pressed)
                elif event.button == 5:
                    handle_zoom(1, ctrl_pressed)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
                    slider_active = False
                    layer_slider_active = False
            elif event.type == MOUSEMOTION:
                if mouse_down:
                    current_pos = event.pos
                    delta_x = current_pos[0] - last_pos[0]
                    delta_y = current_pos[1] - last_pos[1]
                    handle_mouse_movement(delta_x, delta_y)
                    last_pos = current_pos
                elif slider_active:
                    # Update randomization intensity based on mouse position
                    mouse_x = event.pos[0]
                    slider_value = (mouse_x - random_slider_rect.x) / random_slider_rect.width
                    random_intensity = max(0.0, min(1.0, slider_value))
                elif layer_slider_active:
                    # Update layer slider value based on mouse position
                    mouse_x = event.pos[0]
                    slider_value = (mouse_x - layer_slider_rect.x) / layer_slider_rect.width
                    # Assuming layer_count ranges from 3 to 21 and must be odd
                    raw_value = 3 + int(slider_value * (21 - 3))
                    raw_value = max(3, min(21, raw_value))
                    if raw_value % 2 == 0:
                        raw_value += 1 if raw_value < 21 else -1
                    layer_slider_value = raw_value
            elif event.type == KEYDOWN:
                pass  # No longer handling text input

        if randomize:
            # Generate a new random structure with the specified intensity
            cubes = generate_random_structure(layer_count, random_intensity)
            glDeleteBuffers(1, [vertex_vbo])
            glDeleteBuffers(1, [edge_vbo])
            vertex_vbo, vertex_count, edge_vbo, edge_count = create_vbo(cubes)
            randomize = False

        # Update layer_count if it has changed
        if layer_slider_value != layer_count:
            layer_count = layer_slider_value
            camera_distance = layer_count * 4  # Update camera distance
            # Regenerate the current structure with the new layer count
            if randomize:
                cubes = generate_random_structure(layer_count, random_intensity)
            else:
                cubes = generate_diamond_structure(layer_count)
            glDeleteBuffers(1, [vertex_vbo])
            glDeleteBuffers(1, [edge_vbo])
            vertex_vbo, vertex_count, edge_vbo, edge_count = create_vbo(cubes)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, (viewport[0] / viewport[1]), 0.1, 1000.0)

        # Set up the modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Calculate camera position
        cam_x = camera_distance * np.sin(np.radians(camera_angle_y)) * np.cos(np.radians(camera_angle_x))
        cam_y = camera_distance * np.sin(np.radians(camera_angle_x))
        cam_z = camera_distance * np.cos(np.radians(camera_angle_y)) * np.cos(np.radians(camera_angle_x))

        # Look at the center of the structure
        gluLookAt(cam_x, cam_y + (layer_count / 2), cam_z,
                  0, (layer_count / 2), 0,
                  0, 1, 0)

        # Draw the structure
        draw_vbo(vertex_vbo, vertex_count, edge_vbo, edge_count)

        # Draw the GUI elements
        draw_gui()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()