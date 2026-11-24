import unittest
import numpy as np
import torch

from qwen_drive_world_model.build_instance_motion import associate, associate_dense, track_dense_boxes
from qwen_drive_world_model.instance_motion_model import history_features,InstanceMotionPredictor,decoded_boxes


class InstanceMotionTest(unittest.TestCase):
    def test_constant_velocity_and_rotation(self):
        h=torch.tensor([[[-4.5,0.,4.,2.,0.],[-3.,0.,4.,2.,0.],[-1.5,0.,4.,2.,0.],[0.,0.,4.,2.,0.]]])
        mask=torch.ones(1,4);category=torch.zeros(1,dtype=torch.long)
        features,bases,omega,current=history_features(h,mask,category)
        self.assertEqual(features.shape,(1,43))
        self.assertTrue(torch.allclose(bases,torch.tensor([[[3.,0.]]]).expand_as(bases),atol=1e-5))
        model=InstanceMotionPredictor()
        prediction=model(features,bases,omega)
        box=decoded_boxes(current,*prediction[:3])
        self.assertAlmostEqual(float(box[0,0].detach()),3.,places=4)
        rotated=h.clone();rotated[:,:,0]=-h[:,:,1];rotated[:,:,1]=h[:,:,0];rotated[:,:,4]+=torch.pi/2
        f2,b2,o2,c2=history_features(rotated,mask,category)
        self.assertTrue(torch.allclose(features,f2,atol=1e-5))
        box2=decoded_boxes(c2,*model(f2,b2,o2)[:3])
        self.assertAlmostEqual(float(box2[0,1].detach()),3.,places=4)

    def test_missing_history_does_not_create_nonfinite_features(self):
        h=torch.zeros(2,4,5);h[:,-1]=torch.tensor([100.,2.,4.,2.,.2])
        mask=torch.zeros(2,4);mask[:,-1]=1
        f,b,o,c=history_features(h,mask,torch.tensor([0,1]))
        self.assertTrue(torch.isfinite(f).all())
        self.assertTrue(torch.equal(b,torch.zeros_like(b)))

    def test_numeric_track_ids_do_not_change_geometric_association(self):
        now=np.array([[0.,0.,4.,2.,0.],[8.,3.,4.2,2.,np.pi]],np.float32)
        labels=np.array([0,0]);ids=np.array([10,20])
        frames=[]
        for dt in [-1.5,-1.,-.5,0.]:
            boxes=now.copy();boxes[0,0]+=dt*3;boxes[1,0]-=dt*2
            frames.append((boxes,labels,ids,np.zeros((2,2))))
        a=associate(now,labels,ids,frames)
        changed=[(b,l,i[::-1],v) for b,l,i,v in frames]
        other=associate(now,labels,ids+100,changed)
        self.assertTrue(np.array_equal(a[0],other[0]))
        self.assertTrue(np.array_equal(a[1],other[1]))

    def test_dense_association_ignores_ids_and_gt_velocity(self):
        now=np.array([[0.,0.,4.,2.,0.],[8.,3.,4.2,2.,np.pi]],np.float32)
        labels=np.array([0,0]);ids=np.array([10,20]);frames=[]
        for step in range(-15,1):
            boxes=now.copy();boxes[0,0]+=step*.1*3;boxes[1,0]-=step*.1*2
            frames.append((boxes,labels,ids,np.zeros((2,2))))
        first=associate_dense(now,labels,ids,frames)
        pure_history,pure_mask=track_dense_boxes(now,labels,[(b,l) for b,l,i,v in frames])
        self.assertTrue(np.array_equal(first[0],pure_history))
        self.assertTrue(np.array_equal(first[1],pure_mask))
        changed=[(b,l,i[::-1],np.ones((2,2))*9999) for b,l,i,v in frames]
        second=associate_dense(now,labels,ids+100,changed)
        self.assertTrue(np.array_equal(first[0],second[0]))
        self.assertTrue(np.array_equal(first[1],second[1]))


if __name__=='__main__':unittest.main()
