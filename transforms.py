"""
HW4: Transforms
Comp 630 W'16 - Computer Graphics
Phillips Academy
2015-1-19

By Jenny Huang
"""
import numpy as np

def translate(v):
  """Returns the 4x4 matrix for a translation.

  "v" is a 1-D, 3-element numpy array, the vector by which to translate.
  """
  translate = np.concatenate((np.eye(3), v.reshape(-1,1)), 1)
  translate = np.concatenate((translate, np.array([[0,0,0,1]])),0)
  return translate

def rotate(angles):
  """Returns the 4x4 matrix for a 3-D rotation.

  "angles" is a 1-D, 3-element numpy array of the yaw/pitch/roll angles, in
  radians.  We consider these angles to define a rotation in the following
  way (intrinsic Z-X-Y angles):
    1. The object (or space) is first yawed around its Z axis.
    2. It is then pitched around the newly-yawed X axis.
    3. Finally, it is rolled around the yawed-and-pitched Y axis.
  """
  zAngle = angles[0]
  xAngle = angles[1]
  yAngle = angles[2]
  rZ = np.array([[np.cos(zAngle), -np.sin(zAngle), 0, 0],
                 [np.sin(zAngle), np.cos(zAngle), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
  rX = np.array([[1, 0, 0, 0],
                 [0, np.cos(xAngle), -np.sin(xAngle), 0],
                 [0, np.sin(xAngle), np.cos(xAngle), 0],
                 [0, 0, 0, 1]])
  rY = np.array([[np.cos(yAngle), 0, np.sin(yAngle), 0],
                 [0, 1, 0, 0],
                 [-np.sin(yAngle), 0, np.cos(yAngle), 0],
                 [0, 0, 0, 1]])
  rotate = rZ.dot(rX.dot(rY))
  return rotate

def scale(factors):
  """Returns the 4x4 matrix for a 3-D scaling transform.

  "factors" is a 1-D, 3-element numpy array, the X/Y/Z scale factors.
  """
  xFac = factors[0]
  yFac = factors[1]
  zFac = factors[2]
  scale = np.array([[xFac, 0, 0, 0],
                    [0, yFac, 0, 0],
                    [0, 0, zFac, 0],
                    [0, 0, 0, 1]])
  return scale

def shear(shear_dim, contrib_dim, factor):
  """Returns the 4x4 matrix for a 3-D shearing transform.

  Arguments:
   - shear_dim: the dimension in which the shear applies.
   - contrib_dim: the dimension that modulates the amount of shear.
   - factor (scalar): the scaling factor for the amount of shear.
  The dimension arguments should each be 0, 1, or 2, which indicate
  X, Y, or Z respectively.
  """
  shear = np.eye(4)
  shear[shear_dim, contrib_dim] = factor
  return shear
