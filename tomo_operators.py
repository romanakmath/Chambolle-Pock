#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD 2-clause Simplified
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from __future__ import division
import numpy as np
import astra

class AstraToolbox:

    def __init__(self, n_pixels, n_angles, rayperdetec=None):
        '''
        Initialize the ASTRA toolbox with a simple parallel configuration.
        The image is assumed to be square, and the detector count is equal to the number of rows/columns.
        '''
        self.vol_geom = astra.create_vol_geom(n_pixels, n_pixels)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, n_pixels, np.linspace(0,np.pi,n_angles,False))
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        #~ self.rec_id = astra.data2d.create('-vol', self.vol_geom)

    def backproj(self, sino_data, filt=False):
        if filt is True:
            bid, rec = astra.create_backprojection(self.filter_projections(sino_data), self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        else:
            bid, rec = astra.create_backprojection(sino_data, self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        astra.data2d.delete(bid)
        return rec

    def proj(self, slice_data):
        sid, proj_data = astra.create_sino(slice_data, self.proj_id)
        astra.data2d.delete(sid)
        return proj_data

    def filter_projections(self, proj_set):
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real

    def cleanup(self):
        #~ astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.proj_id)
        
        
        
class AstraToolbox2:

    def __init__(self, n_pixels, n_angles, rayperdetec=None):
        '''
        Initialize the ASTRA toolbox with a simple parallel configuration.
        The image is assumed to be square, and the detector count is equal to the number of rows/columns.
        '''
        
        #~ self.rec_id = astra.data2d.create('-vol', self.vol_geom)
        
        
        vox_res = 256

        scene_width = 127.5
        self.vol_geom = astra.create_vol_geom(vox_res, vox_res, -scene_width, scene_width, -scene_width, scene_width)
        
        
        #Length diagonal is half of the length of orthogonal
        #Scene width in cm
        b=scene_width*2
        # Distance from centre do diagonal source array
        d=np.sqrt ( 2 * b**2 )
        #Diagonal source array width which is assumed to be the half of the horizontal source array width
        det_width = d/(np.sqrt(2)+1/2)
        # Squared length of the line connecting the centres of the horizontal and diagonal detectors, general pythagoras for cosine
        fsquar=det_width**2*(5+2*np.sqrt(2))/4
        # Height from the boundary of the scene to the sourcearray 
        h = np.sqrt(fsquar - b**2)
        # Orthogonal source to origin distance
        osod = b+h
        
        print("scene width: "+ str(b) + " cm")
        print(d)
        print(det_width)
        print(osod)
        
        # We generate the geometry for the setup with three different directions. 
        # snum determines the number of sources in each plane of the sources and angles
        # the directions in which the rays are taken
        # To just see the projection directions set sources number to 1
        snum = 27
        angles = np.array ( [0,np.pi /4, np.pi/2] )
        #angles = np.linspace(0, 2*np.pi, 50, False)
        vectors = np.zeros((len(angles)*snum, 6))
        pix_num_half = 1024
        sor_spacing=det_width/(snum-1)
        
        
        
        for j,angle in enumerate(angles):
            for i in range(snum):
                #translation in the detector plane for each new source
                transl = np.array( [  np.cos(angle) , np.sin(angle) ] )*sor_spacing*(i- (snum-1)/2) 
                # source
                if not np.isclose(angle, np.pi/4):
                    vectors[j*snum+i,0:2] = np.array( [ np.sin(angle),  - np.cos(angle)] ) * osod + 2* transl 
                    
                else :
                    vectors[j*snum+i,0:2] = np.array( [ np.sin(angle),  - np.cos(angle)] ) *d + transl
                # center of detector
                if not np.isclose(angle, np.pi/4):
                    vectors[j*snum+i,2:4] = np.array( [ -np.sin(angle),  np.cos(angle)] ) * osod + 2* transl
                    
                else :
                    vectors[j*snum+i,2:4] = np.array( [- np.sin(angle),  np.cos(angle)] ) *d + transl
             
              # vector from detector pixel (0,0) to (0,1)
              # x*2 for orthogonal detectors
                if not np.isclose(angle, np.pi/4):
                    vectors[j*snum+i,4:6] =  np.array( [  np.cos(angle) , np.sin(angle) ] ) *  det_width/pix_num_half
                else : 
                    vectors[j*snum+i,4:6] = np.array( [  np.cos(angle) , np.sin(angle) ] ) * det_width/2 /pix_num_half
                    
        
        self.proj_geom = astra.create_proj_geom('fanflat_vec', 2*pix_num_half, vectors)
        # As before, create a sinogram from a phantom
        self.proj_id = astra.create_projector('line_fanflat',self.proj_geom,self.vol_geom)

    def backproj(self, sino_data, filt=False):
        if filt is True:
            bid, rec = astra.create_backprojection(self.filter_projections(sino_data), self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        else:
            bid, rec = astra.create_backprojection(sino_data, self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        astra.data2d.delete(bid)
        return rec

    def proj(self, slice_data):
        sid, proj_data = astra.create_sino(slice_data, self.proj_id)
        astra.data2d.delete(sid)
        return proj_data

    def filter_projections(self, proj_set):
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real

    def cleanup(self):
        #~ astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.proj_id)

class AstraToolboxVolPar:

    def __init__(self, n_pixels, n_angles, rayperdetec=None):
        '''
        Initialize the ASTRA toolbox with a simple parallel configuration.
        The image is assumed to be square, and the detector count is equal to the number of rows/columns.
        '''
        
        self.vol_geom = astra.create_vol_geom(128, 128, 128)
        angles = np.linspace(0, np.pi, 180,False)
        self.proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 192, angles)
        self.proj_id = astra.create_projector('cuda3d',self.proj_geom,self.vol_geom) 

    def backproj(self, sino_data, filt=False):
        if filt is True:
            bid, rec = astra.create_backprojection(self.filter_projections(sino_data), self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        else:
            bid, rec = astra.create_backprojection(sino_data, self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        astra.data2d.delete(bid)
        return rec
    
    def proj(self, slice_data):
        sid, proj_data = astra.create_sino(slice_data, self.proj_id)
        astra.data2d.delete(sid)
        return proj_data
        
    def run_algorithm(self, alg, n_it, data):
        rec_id = astra.data3d.create('-vol', self.vol_geom)
        #sino_id = astra.data2d.create('-sino', self.proj_geom, data)
        sino_id = astra.create_sino3d_gpu(data, self.proj_geom, self.vol_geom, returnData=False)     
        cfg = astra.astra_dict(alg)
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        alg_id = astra.algorithm.create(cfg)
        print("Running %s" %alg)
        astra.algorithm.run(alg_id, n_it)
        rec = astra.data3d.get(rec_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
        return rec
        
    def filter_projections(self, proj_set):
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real

    def cleanup(self):
        #~ astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.proj_id)


class AstraToolboxVolCon:

    def __init__(self, det_row_count=257,det_col_count=None, n_proj=32, ctpara=None):
        '''
        Initialize the ASTRA toolbox with a simple parallel configuration.
        The image is assumed to be square, and the detector count is equal to the number of rows/columns.
        '''
        
        
        vox_res = 128
        scene_width = float(ctpara['VoxelSizeX'])*1000*128/2 *10
        print("scene_width in one direction is: " + str(scene_width) )
        self.vol_geom = astra.create_vol_geom(vox_res, vox_res, vox_res,-scene_width,scene_width,-scene_width,scene_width,-scene_width,scene_width)
        
        #Angle setting
        print("Angle parameters")
        
        InitialAngle=float(ctpara['InitialAngle'])
        AngularStep=float(ctpara['AngularStep'])
        n_proj_total = int(ctpara['Projections'])
        print("IA " + str(InitialAngle) + " AngularStep " +str(AngularStep))
        
        
        self.angles,step = np.linspace(InitialAngle/360*2*np.pi,InitialAngle/360*2* np.pi+ (n_proj_total)*AngularStep/360*2*np.pi, n_proj,False,retstep= True)
        self.angles *=-1
        print("Angular step: " + str(AngularStep/360*2*np.pi*n_proj_total/n_proj))
        print("Linspace step: " + str(step))
        print("Size of array: " + str(self.angles.size))
        #Measurement in meter 127 10 -6
        self.det_spacing_x = 127
        self.det_spacing_y = 127
        
        self.det_row_count = det_row_count
        if det_col_count==None:
            self.det_col_count = det_row_count
        print("Creating reconstruction geometry with")
        print(str(self.det_row_count) + "rows and " + str(self.det_col_count) + "cols.")
        
        
        
        self.source_origin = float(ctpara['SrcToObject'])*1200
        print("Source origin distance in micrometer: " + str(self.source_origin) )
        self.origin_det = float(ctpara['SrcToDetector'])*1000 - self.source_origin
        print("Origin detector distance in micrometer: " + str(self.origin_det) )
        
        self.proj_geom = astra.create_proj_geom('cone', self.det_spacing_x, self.det_spacing_y , self.det_row_count, self.det_col_count, self.angles,\
                                                self.source_origin, self.origin_det)
        self.proj_id = astra.create_projector('cuda3d',self.proj_geom,self.vol_geom) 
       

    def backproj(self, sino_data, filt=False):
        if filt is True:
            bid, rec = astra.create_backprojection(self.filter_projections(sino_data), self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        else:
            bid, rec = astra.create_backprojection(sino_data, self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        astra.data2d.delete(bid)
        return rec
    
    def proj(self, slice_data):
        sid, proj_data = astra.create_sino(slice_data, self.proj_id)
        astra.data2d.delete(sid)
        return proj_data
        
    def run_algorithm(self, alg, n_it, data,proj=False):
        rec_id = astra.data3d.create('-vol', self.vol_geom)
        #sino_id = astra.data2d.create('-sino', self.proj_geom, data)
        if proj == False:
            sino_data,sino_id = astra.create_sino3d_gpu(data, self.proj_geom, self.vol_geom, returnData=True)  
        else:
           
            sino_data = astra.data3d.create('-sino', self.proj_geom, data)
            #sino_data = data
        print("Sino data shape:")
        #print(sino_data.shape)
        cfg = astra.astra_dict(alg)
        
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_data
        
        alg_id = astra.algorithm.create(cfg)
        print("Running %s" %alg)
        astra.algorithm.run(alg_id, n_it)
        rec = astra.data3d.get(rec_id)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sino_data)
        return rec
        
    def filter_projections(self, proj_set):
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real

    def cleanup(self):
        #~ astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.proj_id)
        

class AstraToolboxVolConVec:

    def __init__(self, det_row_count=2304,det_col_count=3200, n_proj=32, ctpara=None):
        '''
        Initialize the ASTRA toolbox with a simple parallel configuration.
        The image is assumed to be square, and the detector count is equal to the number of rows/columns.
        '''
  
        vox_num_x = int(ctpara['VoxelsX']) 
        vox_num_y = int(ctpara['VoxelsY']) 
        vox_num_z = int(ctpara['VoxelsZ']) 
        #Determine scene width
        sw_x = float(ctpara['VoxelSizeX'])* 3200/det_col_count*vox_num_x *3
        sw_y = float(ctpara['VoxelSizeY'])* 3200/det_col_count*vox_num_y *3
        sw_z = float(ctpara['VoxelSizeZ'])*2304/det_row_count *vox_num_z *3
        #Make proper number of rows and cols
        
        
        
        #=======================================================================
        # sw_x = float(ctpara['VoxelSizeX'])*det_col_count*3
        # sw_y = float(ctpara['VoxelSizeY'])*det_col_count*3
        # sw_z = float(ctpara['VoxelSizeZ'])*det_row_count*3
        #=======================================================================
        
        print("The Voxel number is:")
        print(str(vox_num_x) + " in x direction")
        print(str(vox_num_y) + " in y direction")
        print(str(vox_num_z) + " in z direction")
        print("The scene width (in mm) is:")
        print(str(sw_x) + " in x direction")
        print(str(sw_y) + " in y direction")
        print(str(sw_z) + " in z direction")
        
        print("Number of projections: " + str(n_proj) )
        
        
        #TO DO Check the proper order of three resolution variables in the volume!
        #self.vol_geom = astra.create_vol_geom(vox_num_x, vox_num_y, vox_num_z,-sw_x,sw_x,-sw_y,sw_y,-sw_z,sw_z)
        self.vol_geom = astra.create_vol_geom(vox_num_x, vox_num_y, vox_num_z,-sw_x,sw_x,-sw_y,sw_y,-sw_z,sw_z)
        #width = 500.5*float(ctpara['VoxelSizeZ'])
        #self.vol_geom = astra.create_vol_geom(vox_res, vox_res, vox_res,-width,width,-width,width,-width,width)
        
        #Angle setting
        print("Angle parameters")
        InitialAngle=float(ctpara['InitialAngle'])
        AngularStep=float(ctpara['AngularStep'])
        n_proj_total = int(ctpara['Projections'])
        print("IA " + str(InitialAngle) + " AngularStep " +str(AngularStep))
        
        
        self.angles,step = np.linspace(InitialAngle/360*2*np.pi,InitialAngle/360*2* np.pi+ (n_proj_total)*AngularStep/360*2*np.pi, n_proj,False,retstep= True)
        self.angles *=-1
        print("Angular step: " + str(AngularStep/360*2*np.pi*n_proj_total/n_proj))
        print("Linspace step: " + str(step))
        print("Size of array: " + str(self.angles.size))
        
        
        self.det_row_count = det_row_count
        if det_col_count==None:
            self.det_col_count = det_row_count
        else:
            self.det_col_count = det_col_count
        
        print("Creating reconstruction geometry with")
        print(str(self.det_row_count) + "rows and " + str(self.det_col_count) + "cols.")
          
        self.source_origin = float(ctpara['SrcToObject'])
        print("Source origin distance in micrometer: " + str(self.source_origin) )
        self.origin_det = float(ctpara['SrcToDetector']) - self.source_origin
        print("Origin detector distance in micrometer: " + str(self.origin_det) )
        vectors = np.zeros((len(self.angles), 12))
        for i,angle in enumerate(self.angles):
        
            # source
            vectors[i,0:3] = np.array( [ np.sin(angle),  - np.cos(angle), 0] ) * self.source_origin 
          
            # center of detector
            vectors[i,3:6] = np.array( [ -np.sin(angle),  np.cos(angle), 0] ) * self.origin_det
            
            # vector from detector pixel (0,0) to (0,1) (x,y,z)
            vectors[i,6:9] =  np.array( [np.cos(angle),np.sin(angle), 0 ] ) * float( ctpara['DetectorPixelSizeX']) * 3200/det_col_count
            
            # vector from detector pixel (0,0) to (1,0)
            vectors[i,9:12] = np.array( [0,0,1] ) * float( ctpara['DetectorPixelSizeY'] ) *2304/det_row_count 
        
        self.proj_geom = astra.create_proj_geom('cone_vec', self.det_col_count, self.det_row_count, vectors)
        self.proj_id = astra.create_projector('cuda3d',self.proj_geom,self.vol_geom) 
        
        

    def backproj(self, data, filt=False):
        
        sino_data = astra.data3d.create('-sino', self.proj_geom,data)
        
        if filt is True:
            bid, rec = astra.create_backprojection3d_gpu(self.filter_projections(sino_data), self.proj_geom, self.vol_geom)#,useCUDA=True) #last keyword for astra 1.1
        else:
            bid, rec = astra.create_backprojection3d_gpu(sino_data, self.proj_geom, self.vol_geom)#,useCUDA=True) #last keyword for astra 1.1
        astra.data3d.delete(bid)
        return rec
    
    def proj(self, slice_data):
        sid, proj_data = astra.create_sino3d_gpu(slice_data, self.proj_geom,self.vol_geom)
        astra.data3d.delete(sid)
        return proj_data
        
    def run_algorithm(self, alg, n_it, data,proj=False):
        rec_id = astra.data3d.create('-vol', self.vol_geom)
        #sino_id = astra.data2d.create('-sino', self.proj_geom, data)
        if proj == False:
            sino_data,sino_id = astra.create_sino3d_gpu(data, self.proj_geom, self.vol_geom, returnData=True)  
        else:      
            sino_data = astra.data3d.create('-sino', self.proj_geom, data)
            #sino_data = data
        print("Sino data shape:")
        #print(sino_data.shape)
        cfg = astra.astra_dict(alg)
        
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_data
        
        alg_id = astra.algorithm.create(cfg)
        print("Running %s" %alg)
        astra.algorithm.run(alg_id, n_it)
        rec = astra.data3d.get(rec_id)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sino_data)
        return rec
        
    def filter_projections(self, proj_set):
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real
    
    #design Filter
    
    def filter_projections(self, proj_set):
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real
    
    def cleanup(self):
        #~ astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.proj_id)
        
       