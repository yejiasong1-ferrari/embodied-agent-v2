import unittest
import numpy as np

from config import AgentConfig
from world.object_memory import ObjectMemory
from world.world_model import WorldObject


def observation(position,category):
    return WorldObject('',f'red {category}','red',category,np.array(position,float),
        np.array([0,0,0,1.]),np.array([.06,.06,.08]) if category=='cup' else np.array([.04,.02,.04]),
        .95,1.)


class HeldFragmentTrackingTests(unittest.TestCase):
    def test_adjacent_fragment_requires_correlated_motion(self):
        memory=ObjectMemory(AgentConfig().perception)
        source=observation([.3,0,.2],'cup');source.id='source'
        fragment=observation([.37,0,.2],'bottle');fragment.id='fragment'
        memory.objects.update(source=source,fragment=fragment)
        memory.held_id='source'
        memory.held_expected_position=np.array([.3,0,.2])
        memory.update([observation([.3,0,.2],'cup'),observation([.37,0,.2],'bottle')],2.)
        self.assertFalse(memory.held_fragments)
        memory.held_expected_position=np.array([.34,0,.2])
        memory.update([observation([.34,0,.2],'cup'),observation([.37,0,.2],'bottle')],3.)
        self.assertFalse(memory.held_fragments)
        memory.held_expected_position=np.array([.38,0,.2])
        memory.update([observation([.38,0,.2],'cup'),observation([.44,0,.2],'bottle')],4.)
        self.assertFalse(memory.held_fragments)
        memory.held_expected_position=np.array([.42,0,.2])
        memory.update([observation([.42,0,.2],'cup'),observation([.48,0,.2],'bottle')],5.)
        self.assertIn('fragment',memory.held_fragments)
        memory.held_id=None
        memory.update([],6.)
        self.assertFalse(memory.held_fragments)


if __name__=='__main__':unittest.main()
