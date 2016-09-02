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
from gfx_helper_mayavi import *

def main():
  fig = setUpFigure()

  # Create a scenegraph with desired values.
  r = makeScenegraph(1, 5, 5, 2,
                    np.array([0, 0, np.pi/4]),
                    np.array([-np.pi/2, 0, 0]),
                    np.array([0, np.pi/2, np.pi/4]))

  # Build a list of shapes in the world frame.
  objs = r.getCompositeTransforms()
  for obj in objs:
      print obj;
      (verts, tris) = obj[1].mesh
      verts = obj[0].dot(verts)
      drawTriMesh(verts[:3,:], tris, fig, edges=False, normals=False)

  showFigure(fig)


def makeScenegraph(diameter, upper_arm, forearm, hand, shoulder, elbow, wrist):
  """Returns the root node of the scenegraph for a robotic-arm scene.

  The arm is composed of three segments: the upper arm, forearm, and hand.
  Each of these has a square cross-section.

  The seven arguments are as follows:
   - diameter: the length of each side of the arm's square cross-section.
   - upper_arm, forearm, hand: the lengths of the segments.
   - shoulder, elbow, wrist: each one is a 1-D 3-element numpy array of
                             [yaw, pitch, roll] angles, in radians.

  For more details, see the assignment.
  """
  # Root node.
  r = sg.RootNode()

  # Rotate to correct orientation.
  rFinal = sg.RotateNode(np.array([0, np.pi/2.0, 0]), "final rotation")
  r.addChild(rFinal)

  # Group of four cubes.
  g_1 = sg.GroupNode("floor, upper arm, forearm, and hand group")
  rFinal.addChild(g_1)

  # Floor.
  t_2 = sg.TranslateNode(np.array([0, -0.5, 0]), "floor trans")
  g_1.addChild(t_2)
  s_1 = sg.ScaleNode(np.array([15,1,15]), "floor scale")
  t_2.addChild(s_1)

  # Group of upper arm, forearm, and hand.
  r_1 = sg.RotateNode(shoulder, "shoulder rotation")
  g_1.addChild(r_1)
  g_2 = sg.GroupNode("upper arm, forearm, and hand group")
  r_1.addChild(g_2)

  # Upper arm.
  s_2 = sg.ScaleNode(np.array([diameter, upper_arm, diameter]),
      "upper arm scale")
  g_2.addChild(s_2)

  # Group of forearm and hand.
  t_3 = sg.TranslateNode(np.array([0, upper_arm*2.0, 0]),
      "forearm and hand translation")
  g_2.addChild(t_3)
  r_2 = sg.RotateNode(elbow, "elbow rotation")
  t_3.addChild(r_2)
  g_3 = sg.GroupNode("forearm and hand group")
  r_2.addChild(g_3)

  # Forearm.
  s_3 = sg.ScaleNode(np.array([diameter,forearm, diameter]),
      "forearm scale")
  g_3.addChild(s_3)

  # Hand.
  t_4 = sg.TranslateNode(np.array([0, 2.0*forearm, 0]),
      "hand translation")
  g_3.addChild(t_4)
  r_3 = sg.RotateNode(wrist, "wrist rotation")
  t_4.addChild(r_3)
  s_4 = sg.ScaleNode(np.array([diameter, hand, diameter]),
      "hand scale")
  r_3.addChild(s_4)

  # First translation of arm up to origin.
  tFirst = sg.TranslateNode(np.array([0, 1, 0]), "arm translation")
  s_2.addChild(tFirst)
  s_3.addChild(tFirst)
  s_4.addChild(tFirst)

  # Add cube shape node.
  c = sg.ShapeNode(meshes.cube(), "cube")
  s_1.addChild(c)
  tFirst.addChild(c)

  return r


if __name__ == "__main__":
  main()
