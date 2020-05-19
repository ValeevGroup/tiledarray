#  This file is a part of TiledArray.
#  Copyright (C) 2020  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import tiledarray as ta

class TiledArrayTest(unittest.TestCase):

  def test_array(self):
    world = ta.get_default_world()
    a = ta.TArray([4,4],2,world)
    a.fill(1, False)
    world.fence()
    a = ta.TArray([5,5],3,world)
    self.assertEqual(a.shape, (5,5))
    a = ta.TArray([[2,7],[2,4,7]])
    self.assertEqual(a.shape, (5,5))
    world.fence()


  def test_expressions(self):
    world = ta.get_default_world()
    a = ta.TArray([4,8],  block=2, world=world)
    b = ta.TArray([8,4], block=2, world=world)
    c = ta.TArray([8,8], block=2, world=world)
    a.fill(1, False)
    c.fill(5, False)
    b["i,j"] = a["j,i"]/2.0
    b["i,j"] = a["j,i"] - a["j,i"]/2 + 6*a["j,i"]*2
    self.assertAlmostEqual((c["i,j"] + c["j,i"]).norm(), 80)
    self.assertAlmostEqual((c["i,j"] + c["j,i"]).min(), 10)
    self.assertAlmostEqual((c["i,j"] - c["j,i"]).max(), 0.0)
    self.assertAlmostEqual(c["i,j"].dot(c["j,i"]), (5**2)*(8*8))
    world.fence()
    del a,b

  def test_einsum(self):
    a = ta.TArray([4,8],  block=2)
    b = ta.TArray([8,12], block=2)
    c = ta.TArray([4,12], block=2)
    a.fill(1, False)
    b.fill(1, False)
    ta.einsum("ik,kj->ij", a, b, c)
    d = ta.TArray()
    ta.einsum("ik,kj,ab->ijab", a, b, c, d)
    ta.get_default_world().fence()

  def test_tile_setitem_getitem(self):
    import numpy as np
    world = ta.get_default_world()
    a = ta.TArray([4,8],  block=2, world=world)
    if world.rank == 0:
      a[0,0] = np.ones([2,2])
    world.fence()
    #print (a[0,0])
    self.assertTrue((a[0,0] == np.ones([2,2])).all())
    world.fence()

  def test_tile_ops(self):
    import numpy as np
    world = ta.get_default_world()
    op=lambda r: np.random.rand(*r.shape)
    a = ta.TArray([4,8],  block=2, world=world, op=op)
    a = ta.TArray([4,8],  block=2, world=world)
    a.init(op)
    print (a[0,0])
    # world.fence
    # self.assertTrue((a[0,0] == np.ones([2,2])).all())
    world.fence()

  def test_array_iter(self):
    import numpy as np
    world = ta.get_default_world()
    a = ta.TArray([4,8],  block=2, world=world)
    #a.fill(0, False)
    for tile in a:
      tile.data = np.ones([2,2])
      print (tile.index, tile.range, tile.data)
    world.fence()

  def test_array_buffer(self):
    import numpy as np
    world = ta.get_default_world()
    a = ta.TArray([4,8],  block=8, world=world)
    a.fill(1)
    world.fence()
    b = np.array(a)
    self.assertEqual(b.shape, a.shape)
    #print (b[...])

if __name__ == '__main__':
  unittest.main()
  ta.get_default_world().fence()
  #ta.finalize()
