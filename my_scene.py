"""
HW4: Transforms
Comp 630 W'16 - Computer Graphics
Phillips Academy
2015-1-19

By Jenny Huang
"""
import numpy as np
import scenegraph as sg
import meshes
import our_shapes as os
from gfx_helper_mayavi import *

def main():
   fig = setUpFigure()

   # Create a scenegraph with desired values.
   r = makeScenegraph(10, 6,
                      np.array([0, np.pi/10, 0]),
                      np.array([0, -np.pi/5, 0]),)

   # Build a list of shapes in the world frame.
   objs = r.getCompositeTransforms()
   for obj in objs:
       (verts, tris) = obj[1].mesh
       verts = obj[0].dot(verts)
       drawTriMesh(verts[:3,:], tris, fig, edges=False, normals=False)

   showFigure(fig)

def makeScenegraph(treeHeight, ballRadius, treeBend, trunkBend):
  """Returns the root node of the scenegraph for a trees-and-ball scene.

  At the center, there is a squished ball. There are two trees, one on either
  side of the ball in the X-direction. Each tree is composed of one double cone
  (leaves and branches) on top of a cylinder (trunk). These all lie on a flat
  rectangular prism floor lying in the XY-plane with the origin of the world
  frame at the center of the top face.

  The four arguments are as follows:
   - treeHeight: the height of each tree
   - ballRadius: the radius of the squishedBall
   - treeBend, trunkBend: each one is a 1-D 3-element numpy array of
                          [yaw, pitch, roll] angles, in radians. TreeBend
                          rotates both trees at the roots, and trunkBend
                          rotates both trees at the top of their trunks.
  """
  # Root node.
  root = sg.RootNode()

  # Rotate to correct orientation.
  r_1 = sg.RotateNode(np.array([0, np.pi/2.0, 0]), "final rotate")
  root.addChild(r_1)

  # Group of floor, squished ball, and two trees.
  # One tree = one doubleCone + one cylinder
  g_1 = sg.GroupNode("every shape group")
  r_1.addChild(g_1)

  # Floor.
  t_1 = sg.TranslateNode(np.array([0, -0.5, 0]), "floor trans")
  g_1.addChild(t_1)
  s_1 = sg.ScaleNode(np.array([15, 1, 15]), "floor scale")
  t_1.addChild(s_1)
  cube = sg.ShapeNode(meshes.cube(), "cube")
  s_1.addChild(cube)

  # SquishedBall.
  s_2 = sg.ScaleNode(np.array([ballRadius, ballRadius, ballRadius]),
      "ball scale")
  g_1.addChild(s_2)
  t_2 = sg.TranslateNode(np.array([0, 1, 0]), "above floor trans")
  s_2.addChild(t_2)
  s = os.squishedBall(10)
  s = (os.vertsToHomogeneous(s[0]), s[1])
  squishedBall = sg.ShapeNode(s, "squished ball")
  t_2.addChild(squishedBall)

  # Two trees.
  r_2 = sg.RotateNode(treeBend, "trees rotate")
  g_1.addChild(r_2)

  # Group of two trees.
  g_3 = sg.GroupNode("two trees")
  r_2.addChild(g_3)

  # Tree one.
  t_3 = sg.TranslateNode(np.array([8, 0, 0]), "tree one trans")
  g_3.addChild(t_3)

  # Tree two.
  t_4 = sg.TranslateNode(np.array([-8, 0, 0]), "tree one trans")
  g_3.addChild(t_4)

  # Group of tree one.
  g_4 = sg.GroupNode("tree one group")
  t_3.addChild(g_4)

  # Group of tree two.
  g_5 = sg.GroupNode("tree two group")
  t_4.addChild(g_5)

  # Two doubleCones.
  t_5 = sg.TranslateNode(np.array([0, treeHeight, 0]))
  g_4.addChild(t_5)
  g_5.addChild(t_5)
  r_3 = sg.RotateNode(trunkBend, "doubleCones rotate")
  t_5.addChild(r_3)
  s_4 = sg.ScaleNode(np.array([2, 0.5*treeHeight, 2]), "doubleCones scale")
  r_3.addChild(s_4)
  t_7 = sg.TranslateNode(np.array([0, 1, 0]), "above floor trans")
  s_4.addChild(t_7)
  d = os.doubleCone(15)
  d = (os.vertsToHomogeneous(d[0]), d[1])
  doubleCone = sg.ShapeNode(d, "double cone")
  t_7.addChild(doubleCone)

  # Two cylinders.
  t_6 = sg.ScaleNode(np.array([1, 0.5*treeHeight, 1]), "cylinders scale")
  g_4.addChild(t_6)
  g_5.addChild(t_6)
  t_8 = sg.TranslateNode(np.array([0, 1, 0]), "above floor trans")
  t_6.addChild(t_8)
  cylinder = sg.ShapeNode(meshes.prism(15), "cylinder")
  t_8.addChild(cylinder)

  return root


if __name__ == "__main__":
  main()
