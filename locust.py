import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.collections import LineCollection
import numba
from matplotlib import colors as mcolors
import pandas as pd
from itertools import combinations

import matplotlib.pyplot as plt; plt.rcdefaults()
from scipy import stats
import matplotlib.mlab as mlab
import math
from tqdm import tqdm

class PairwiseInteraction():
    
    def __init__(self, posture_data_location, skeleton):
        self.posture_data_location = posture_data_location
        self.skeleton = skeleton


    def get_data(self, frame, skeleton, posture_data_location):
        """ Export data from h5 file: 'locust_posture_data.h5'

        Parameters
        ----------
        frame : int
            Frame number of which data will be exported from

        skeleton : array-like
            Array containing x,y coordinates, shape: (35, 2)

        Returns
        -------
        allLocusts : list, 
            List of line segments that make up each individual 
            locust, shape: (103, 26, 2, 2)

        totalIndividuals: int
            Number of total locusts in the posture data 
        """
        with h5py.File(self.posture_data_location) as h5file:
            compressed = h5file['joints'][:,frame,...,:2]

        N_INDIVID = len(compressed)
        allLocusts = [] # empty list. fill this will all individual's line segment stacks

        for individ in range(N_INDIVID):
            line_segments = [] # empty list
            for idx, jdx in enumerate(skeleton[:,0]):
                if jdx != -1: # if node has parent... 
                    line_segments.append(compressed[individ][[idx,jdx]])
            allLocusts.append(np.stack(line_segments))  
            # shape of line_segments (26, 2, 2)
            # shape of allLocusts (103, 26, 2, 2)
            totalIndividuals = len(line_segments)
        return allLocusts, totalIndividuals

    # @numba.njit(fastmath=True)
    

    # @numba.njit(fastmath=True)
    def line_line_intersection(self, a, b):
        def det(a, b, c, d):    
            return a*d - b*c

        intersects = False
        intersection = np.empty((1,2))+np.nan

        x1 = a[0,0]
        x2 = a[1,0]
        y1 = a[0,1]
        y2 = a[1,1]

        x3 = b[0,0]
        x4 = b[1,0]
        y3 = b[0,1]
        y4 = b[1,1]

        dxa = x1 - x2
        dxb = x3 - x4

        dya = y1 - y2
        dyb = y3 - y4

        denominator = det(dxa, dya, dxb, dyb)
        parallel = denominator == 0

        if not parallel:

            dyab = y4 - y1
            dxab = x4 - x1
            lamda = ((-dyb) * (dxab) + (dxb) * (dyab)) / denominator
            gamma = ((dya) * (dxab) + (-dxa) * (dyab)) / denominator
            intersects = (0 < lamda < 1) & (0 < gamma < 1)

            ''' # previous method of checking endpoint intersecting limb
            dxc = x1 - x3
            dyc = y1 - y3

            dxl = x4 - x1
            dyl = y4 - y3

            # check if point is on the line
            cross = dxc * dyl - dyc * dxl

            if cross == 0: # point is on line. count this as an intersection
                intersects = True

            # new method of checking endpoint intersecting limb
            if not intersects: # check if limb of locustA contains point of limbB
                # check overlap
                intersects, intersection = check_all_overlap(x1, y1, x2, y2, x3, y3, x4, y4)       
            if intersects: # calculate point of intersection

                det_a = det(x1, y1, x2, y2)
                det_c = det(x3, y3, x4, y4)

                x = det(det_a, dxa, det_c, dxb) / denominator

                y = det(det_a, dya, det_c, dyb) / denominator

                intersection[0,0] = x
                intersection[0,1] = y
        '''
        return intersects, intersection

    # @numba.njit(fastmath=True)
    def get_intersection_matrix(self, line_segments_A, line_segments_B):  
        """ Get intersection matrix between locust A and B

        Parameters
        ----------
        line_segments_A : {array-like, sparse matrix}, shape (26, 2, 2)
            Line segments of locust A

        line_segments_B : {array-like, sparse matrix}, shape (26, 2, 2)
            Line segments of locust B

        Returns
        -------
        intersection_matrix : {array-like, sparse matrix}, shape (26, 26)
            Intersection matrix between Locust A's 26 limbs and B's 26 limbs

        """
        intersection_matrix = np.zeros((26, 26))
        for Adx in range(len(line_segments_A)):
            jointA  = line_segments_A[Adx] # for each joint in locust A
            for Bdx in range(len(line_segments_B)):
                jointB = line_segments_B[Bdx] # for each joint in Locust B
                if (self.line_line_intersection(jointA, jointB)[0]): # if there is an intersection between joints
                    intersection_matrix[Adx, Bdx]=1
                    break
        return intersection_matrix

    # @numba.njit(parallel=True)
    def get_adjmatrix(self, N_INDIVID, N_SEGMENTS, allLocusts):
        """ Get an large  intersection matrix of size (N_INDIVID, N_INDIVID, N_SEGMENTS, N_SEGMENTS) 

        Parameters
        ----------
        N_INDIVID: int
            Number of individual locusts (103)
        N_SEGMENTS: int
            Number of line segments that make up locusts body (26)
        allLocusts : list , shape: (26, 2, 2)
            contains the line segments that make up the joints of all the locusts


        Returns
        -------
        adj_matrix : {array-like, sparse matrix}, shape (103, 103, 26, 26)

        """
        adj_matrix = np.zeros((N_INDIVID, N_INDIVID, N_SEGMENTS, N_SEGMENTS))
        for locustA in numba.prange(N_INDIVID):
        # for locustA in range(N_INDIVID):
            # for locustB in range(N_INDIVID):
            for locustB in numba.prange(N_INDIVID):
                if (locustA != locustB): # only the interaction matrix of locust interacting with other individuals
                    intersection = self.get_intersection_matrix(allLocusts[locustA], allLocusts[locustB])
                    adj_matrix[locustA, locustB, :, :] = intersection  
        return adj_matrix

    # https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap
    def overlapping1D(self, line1, line2):
        min1, max1 = line1[0], line1[1]
        min2, max2 = line2[0], line2[1]
        return (max1>=min2 and max2>=min1)

    def bounding_box_overlap(self, box1, box2):
        in_xrange = self.overlapping1D(box1[0,:], box2[0,:]) # x 
        in_yrange = self.overlapping1D(box1[1,:], box2[1,:]) # y
        return (in_xrange and in_yrange)
   

    def bounding_box(self, line_segs):
            min_x, min_y = np.min(line_segs, axis=0)
            max_x, max_y = np.max(line_segs, axis=0)
            return np.array([(min_x, max_x), (min_y, max_y)])
            # return np.array([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)])
        

    def get_pairwise(self, init_frame, time_window):

        """ Get Time Series Data with shape (N_INDIVID, time_window, segments, segments)
                touching : boolean
                    in x frames back, is LocustA touching LocustB?

        Parameters
        ----------
        init_frame: int
            Initial frame
        time_window: int
            How many frames are you going back


        Returns
        -------
        TS_data : {array-like, sparse matrix}, shape (103, time_window, 26, 26)
            Obtain time series data 

        """
        
        
        TS_data = np.zeros((103, time_window, 26, 26))
        ts_count = 0

        # Calculate Bounding Boxes
        for frame in range(init_frame - time_window + 1, init_frame + 1): # ex: frame 5-10 (inclusive)
            if frame%10==0:
                print("frame: ", frame)

            bounding_boxes=[]
            # allLocusts = data of locust joints # N_SEGMENTS = 26 
            allLocusts, N_SEGMENTS = self.get_data(frame = frame, skeleton = self.skeleton, posture_data_location=self.posture_data_location) 

            # compute adjacency matrix (26, 26) with 0/1 for touching by checking overlapping bounding boxes

            # compute bounding boxes for all
            for locust in range(103):
                bounding_boxes.append(self.bounding_box(allLocusts[locust].reshape(52, 2)))

            for locustA, locustB in combinations(range(103), 2):  
                # if bounding boxes intersect
                if self.bounding_box_overlap(bounding_boxes[locustA], bounding_boxes[locustB]):
                    # get intersection matrix where rows represent LocustA limbs and cols represent locustB limbs
                    intersection_matrix=self.get_intersection_matrix(allLocusts[locustA], allLocusts[locustB])
                    TS_data[locustA, ts_count, :, :]=TS_data[locustA, ts_count, :, :]+intersection_matrix

                    # transpose intersection matrix so rows are now locustB's limbs
                    TS_data[locustB, ts_count, :, :]+=np.transpose(intersection_matrix)

            ts_count += 1
        return TS_data
    
    def bounding_box_plot(self, line_segs):
        min_x, min_y = np.min(line_segs, axis=0)
        max_x, max_y = np.max(line_segs, axis=0)
        # return np.array([(min_x, max_x), (min_y, max_y)])
        return np.array([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)])

    def get_pairwise_vid(self, init_frame, time_window):

        """ Get Time Series Data with shape (N_INDIVID, time_window, segments, segments)
                touching : boolean
                    in x frames back, is LocustA touching LocustB?

        Parameters
        ----------
        init_frame: int
            Initial frame
        time_window: int
            How many frames are you going back


        Returns
        -------
        TS_data : {array-like, sparse matrix}, shape (103, time_window, 26, 26)
            Obtain time series data 

        """

        TS_data = np.zeros((103, time_window, 26, 26))
        plt.style.use('dark_background')
        ts_count = 0

        # Calculate Bounding Boxes
        for frame in range(init_frame - time_window + 1, init_frame + 1): # ex: frame 5-10 (inclusive)
            print("frame", frame)
            bounding_boxes=[]
            # allLocusts = data of locust joints # N_SEGMENTS = 26 
            allLocusts, N_SEGMENTS = self.get_data(frame = frame, skeleton = self.skeleton, posture_data_location=self.posture_data_location) 

            # compute adjacency matrix (26, 26) with 0/1 for touching by checking overlapping bounding boxes

            bbox_overlap=np.zeros(103)

            # compute bounding boxes for all
            for locust in range(103):
                bounding_boxes.append(self.bounding_box(allLocusts[locust].reshape(52, 2)))

            for locustA, locustB in combinations(range(103), 2):  
                # if bounding boxes intersect
                if self.bounding_box_overlap(bounding_boxes[locustA], bounding_boxes[locustB]):
                    bbox_overlap[locustA]=1
                    bbox_overlap[locustB]=1
                    # get intersection matrix where rows represent LocustA limbs and cols represent locustB limbs
                    intersection_matrix=self.get_intersection_matrix(allLocusts[locustA], allLocusts[locustB])
                    TS_data[locustA, ts_count, :, :]+=intersection_matrix

                    # transpose intersection matrix so rows are now locustB's limbs
                    TS_data[locustB, ts_count, :, :]+=np.transpose(intersection_matrix)

            X=np.sum(np.sum(TS_data, axis=3), axis=2)
            touching=np.squeeze(X)

            fig = plt.figure(figsize=(20,20))
            ax = plt.gca()
            for locust in range(103):
                line_collection = LineCollection(allLocusts[locust], colors='b')
                ax.add_collection(line_collection)
                box=self.bounding_box_plot(allLocusts[locust].reshape(52, 2))
                # if bounding box intersects
                if touching[locust][ts_count] > 0:
                    # print('box:', box[:, 0], box[:, 1])
                    plt.plot(box[:, 0], box[:, 1], 'w')
                elif bbox_overlap[locust]==1:
                    plt.plot(box[:, 0], box[:, 1], ':w')

                else: 
                    plt.plot(box[:, 0], box[:, 1], ':b')


            plt.plot(0, 0, 'w', label='Locust with Intersecting Limbs')
            plt.plot(0, 0, ':w', label='Locust with Intersecting Bounding Boxes')
            plt.plot(0, 0, ':b', label='Non-Intersecting Locust')
            plt.legend()
            plt.xlim(80,80+2048)
            plt.ylim(80,80+2048)

            ax.axis('off')

            fig.savefig('./img4video/%d.png' % frame)
            plt.close(fig)

            # plt.show()

            ts_count += 1

        return TS_data

    
    def sample_frame(self, frame):
        TS_data = np.zeros((103, 1, 26, 26))
        ts_count = 0
        print("frame", frame)
        bounding_boxes=[]

        plt.style.use('dark_background')


        # allLocusts = data of locust joints # N_SEGMENTS = 26 
        allLocusts, N_SEGMENTS = self.get_data(frame = frame, skeleton = self.skeleton, posture_data_location=self.posture_data_location) 

        # compute adjacency matrix (26, 26) with 0/1 for touching by checking overlapping bounding boxes
        bbox_overlap=np.zeros(103)

        # compute bounding boxes for all
        for locust in range(103):
            # print(bounding_box(allLocusts[locust]))
            bounding_boxes.append(self.bounding_box(allLocusts[locust].reshape(52, 2)))

        for locustA, locustB in combinations(range(103), 2):  
            # if bounding boxes intersect
            if self.bounding_box_overlap(bounding_boxes[locustA], bounding_boxes[locustB]):
                bbox_overlap[locustA]=1
                bbox_overlap[locustB]=1
                # get intersection matrix where rows represent LocustA limbs and cols represent locustB limbs
                intersection_matrix=self.get_intersection_matrix(allLocusts[locustA], allLocusts[locustB])
                TS_data[locustA, ts_count, :, :]+=intersection_matrix

                # transpose intersection matrix so rows are now locustB's limbs
                TS_data[locustB, ts_count, :, :]+=np.transpose(intersection_matrix)

        X=np.sum(np.sum(TS_data, axis=3), axis=2)
        touching=np.squeeze(X)

        fig = plt.figure(figsize=(10, 20))
        ax = plt.gca()
        for locust in range(103):
            line_collection = LineCollection(allLocusts[locust], colors='c')
            ax.add_collection(line_collection)
            box=self.bounding_box_plot(allLocusts[locust].reshape(52, 2))
            # if bounding box intersects
            if touching[locust] > 0:
                # print('box:', box[:, 0], box[:, 1])
                plt.plot(box[:, 0], box[:, 1], 'w')
            elif bbox_overlap[locust]==1:
                plt.plot(box[:, 0], box[:, 1], ':w')

            else: 
                plt.plot(box[:, 0], box[:, 1], ':b')

        ax.axis('off')

        plt.plot(0, 0, 'w', label='Locust with Intersecting Limbs')
        plt.plot(0, 0, ':w', label='Locust with Intersecting Bounding Boxes')
        plt.plot(0, 0, ':b', label='Non-Intersecting Locust')
        plt.legend()

        plt.xlim(1900,80+2048)
        plt.ylim(800, 1100)

        plt.show()