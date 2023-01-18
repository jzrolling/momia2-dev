from scipy.ndimage import median_filter
from ..classify import prediction2foreground,prediction2seed
from ..utils import generic,skeleton,linalg
from ..metrics import image_feature
from skimage import filters, morphology, measure, feature, segmentation
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
    
class CellTracker:
    
    def __init__(self,
                 sorted_patches, #pre-loaded patches as list
                 time_points,
                 pixel_classifier,
                 seed_prob_min=0.2,
                 edge_prob_max=0.9,
                 background_prob_max=0.2,
                 min_overlap_threshold = 0.5,
                 min_overlap_area=100,
                 min_size_similarity=0.75,
                 min_cell_size=200,
                 min_seed_size=10,
                 max_iter=5,
                 cache_image_features=False,
                 verbose = True,
                 backtrack_generations=4,
                 no_merge=False,
                 no_split=False,
                 hard_split=False):
        
        if len(time_points) != len(sorted_patches):
            raise ValueError("The number of time points given doesn't match the number of image patches!")
        self.timepoints = time_points
        self.frames = sorted_patches
        self.pixel_classifier = pixel_classifier
        self.seed_prob_min = seed_prob_min
        self.edge_prob_max = edge_prob_max
        self.background_prob_max = background_prob_max
        self.min_seed_size = min_seed_size
        self.cache_in = cache_image_features
        
        self.min_overlap_threshold = min_overlap_threshold
        self.min_overlap_area = min_overlap_area
        self.min_size_similarity = min_size_similarity
        self.min_cell_size = min_cell_size
        self.backtrack_generations=backtrack_generations
        self.max_iter = max_iter
        
        self.regionprops = None
        self.overlap_stats = None
        self.to_be_merged = [-1]
        self.to_be_split = [-1]
        self.verbose = verbose
        self.no_merge = no_merge
        self.no_split = no_split
        self.hard_split = hard_split
        
    def run_segmentation(self):
        for i,p in enumerate(self.frames):
            if p.labeled_mask is None:
                if self.verbose:
                    print('Running segmentation on frame {}...'.format(i))
                p.pixel_classification(self.pixel_classifier,
                           seed_prob_min=self.seed_prob_min,
                           edge_prob_max=self.edge_prob_max,
                           background_prob_max=self.background_prob_max,
                           cache_in=self.cache_in,
                           min_seed_size=self.min_seed_size)
        
    def locate_spurious_events(self):
        to_be_split,to_be_merged = _locate_spurious_events(self.regionprops,verbose=self.verbose)        
        self.to_be_split=to_be_split
        self.to_be_merged=to_be_merged
    
    def update_masks(self):
        if len(self.to_be_merged)>0:
            if not self.no_merge:
                for c in self.to_be_merged:
                    self._merge_cells(c)
        if len(self.to_be_split)>0:
            if not self.no_split:
                for c in self.to_be_split:
                    self._split_cells(c)
    
    def update_frames(self):
        _regionprops=[]
        for i,p in enumerate(self.frames):
            p.locate_particles()
            p.regionprops['$cell']=['{}_{}'.format(i,l) for l in p.regionprops.index]
            p.regionprops['$time']=i
            _regionprops.append(p.regionprops)
        self.regionprops = pd.concat(_regionprops).set_index('$cell')
    
    def _update_regionprops(self):
        _regionprops=[]
        for i,p in enumerate(self.frames):
            _regionprops.append(p.regionprops)
            p.regionprops['$cell']=['{}_{}'.format(i,l) for l in p.regionprops.index]
        _regionprops = pd.concat(_regionprops).set_index('$cell')
        for col in _regionprops.columns:
            if col not in self.regionprops.columns:
                self.regionprops[col] = _regionprops.loc[self.regionprops.index][col]
        del _regionprops
            
    def trace_by_overlap(self):
        counter = 0
        while counter<=self.max_iter:
            self.update_frames()
            self.link_cells_by_overlap()
            self.locate_spurious_events()
            self.update_masks()
            if self._end_iter():
                self.update_frames()
                self.link_cells_by_overlap()
                break
            counter+=1
            
    def link_cells_by_overlap(self):
        self.regionprops,self.overlap_stats = overlap_match(self.frames,
                                                             self.regionprops,
                                                             min_cell_size=self.min_cell_size,
                                                             min_size_similarity=self.min_size_similarity,
                                                             min_overlap_threshold=self.min_overlap_threshold,
                                                             min_overlap_area = self.min_overlap_area)
        
    def _end_iter(self):
        end = False
        if (len(self.to_be_merged)+len(self.to_be_split))==0:
            end = True
        return end
    
    def _merge_cells(self, cell_list):

        coords = self.regionprops.loc[np.array(cell_list)]['$coords']
        dilated_coords = [dilate_by_coords(x,self.frames[0].shape,morphology.disk(1)) for x in coords]
        unique_coords,pix_count = np.unique(np.vstack(dilated_coords),return_counts=True,axis=0)
        border_coords = unique_coords[pix_count>=2]
        merged_coords = np.vstack(list(coords)+[border_coords])
        time = np.unique([int(x.split('_')[0]) for x in cell_list])[0]
        new_label = np.min([int(x.split('_')[1]) for x in cell_list])
        self.frames[time].labeled_mask[merged_coords[:,0],merged_coords[:,1]]=new_label
    
    def _split_cells(self, cell_ref):
        cell, n_daughter = cell_ref
        time,label = np.array(cell.split('_')).astype(int)
        x,y = self.regionprops.loc[cell]['$coords'].T
        cropped_prob = np.zeros(self.frames[time].prob_mask.shape)
        cropped_mask = np.zeros(self.frames[time].mask.shape)
        cropped_mask[x,y]=1
        cropped_prob[x,y,:] = self.frames[time].prob_mask[x,y,:]
        
        for th in np.linspace(self.seed_prob_min,1,5):
            seed = prediction2seed(cropped_prob,
                                   seed_min=th,
                                   edge_max=self.edge_prob_max,
                                   min_seed_size=self.min_seed_size)

            if seed.max()>=n_daughter:
                break
                
                
        if seed.max()!= n_daughter and self.hard_split:
            seed = np.zeros(self.frames[time].mask.shape)
            current_mask = self.frames[time].labeled_mask==label
            prev_frame = (self.frames[time-1].labeled_mask)*current_mask
            counter=0
            for d in np.unique(prev_frame): 
                if d!=0:
                    if (np.sum(prev_frame==d)/len(x))>0.1:
                        seed[prev_frame==d]=counter+1            
                        counter+=1
        if seed.max()==n_daughter:
            seed[seed==1]=self.frames[time].labeled_mask.max()+1
            seed[seed==2]=self.frames[time].labeled_mask.max()+2
            cropped_watershed = segmentation.watershed(image=cropped_prob[:,:,1],
                                                       mask=cropped_mask,
                                                       markers=seed,
                                                       connectivity=1,
                                                       compactness=0.1,
                                                       watershed_line=False)
            new_labeled_mask = self.frames[time].labeled_mask.copy()
            new_labeled_mask[x,y]=cropped_watershed[x,y]
            self.frames[time].labeled_mask = new_labeled_mask
        
    def plot_cell(self,cell_id):
        show_highlight(cell_id,self.frames)
        
    def trace_lineage(self):
        rev_lineage = {}
        init_cell_counter = 1
        for c in self.regionprops.index:
            daughters = self.regionprops.loc[c]['daughter(s)']
            if c not in rev_lineage:
                rev_lineage[c]=str(init_cell_counter)
                init_cell_counter+=1
            if len(daughters)==1:
                rev_lineage[daughters[0]]=rev_lineage[c]
            else:
                for i,d in enumerate(daughters):
                    rev_lineage[d] = '{}.{}'.format(rev_lineage[c],i+1)
        lineage = [rev_lineage[c] for c in self.regionprops.index]
        self.regionprops['cell_lineage'] = lineage
        
def dilate_by_coords(coords,image_shape,
                     selem=morphology.disk(2)):
    h,w = image_shape
    if len(selem)==1:
        return coords
    elif len(selem)>1:
        x,y=image_feature.coord2mesh(coords[:,0],
                                     coords[:,1],
                                     selem=selem)
        x[x<0]=0
        x[x>h-1]=h-1
        y[y<0]=0
        y[y>w-1]=w-1
        xy = np.unique(np.array([x.ravel(),y.ravel()]).T,axis=0)
        return xy

def back_track(regionprops,
               cell_id,
               n_generations=3):
    # the backtrack algorithm is based on the naive hypothesis that an fragments of an oversegmented cell
    # should have a common ancester tracing back n generations whereas an undersegmented cell should not.
    
    time,label = np.array(cell_id.split('_')).astype(int)
    elderlist = []
    current_cell = np.array([cell_id])
    while True:
        if n_generations==0 or time ==0:
            break
        mothers = np.unique(np.hstack(regionprops.loc[current_cell]['mother(s)'].values))
        current_cell = mothers
        n_generations-=1
        time -= 1
        if len(mothers)>0:
            elderlist.append(mothers)
        else:
            break
    return elderlist

def area_ratio(areas):
    a1,a2=areas
    if a1>a2:
        ratio = a2/a1
    else:
        ratio = a1/a2
    return ratio

def show_highlight(cell_id,frames):
    t,label = np.array(cell_id.split('_')).astype(int)
    mask = (frames[t].labeled_mask>0)*1
    mask[frames[t].labeled_mask==label]=3
    fig=plt.figure(figsize=(10,10))
    plt.imshow(mask)
    
def overlap_match(frames,regionprops,
                  min_overlap_threshold = 0.5,
                  min_overlap_area=100,
                  min_size_similarity=0.75,
                  min_cell_size=200):
    
    # link cells by overlap
    init_stat = {k:[[],[]] for k in regionprops.index}
    mother_list = []
    daughter_list = []
    rp = regionprops.copy()
    overlap_stat = []
    for t in range(0,len(frames)-1):
        df = _calculate_overlap_matrix(frames[t].labeled_mask,
                                       frames[t+1].labeled_mask,t,t+1)
        filtered_df = df[np.max(df[['frame1_overlap_frac','frame2_overlap_frac']].values,axis=1)>min_overlap_threshold]
        for (cell1,cell2) in filtered_df[['frame1_id','frame2_id']].values:
            init_stat[cell1][1] += [cell2]
            init_stat[cell2][0] += [cell1]
        overlap_stat.append(df)
    
    overlap_stat=pd.concat(overlap_stat)
    
    # find missing mother(s) for cells that moved a bit
    orphans_candidates = rp[(rp['$time']>0)&(rp['$touching_edge']==0)].index
    for o in orphans_candidates:
        if len(init_stat[o][0])==0 and len(init_stat[o][1])>0:
            subset = overlap_stat[overlap_stat['frame2_id']==o].values
            missing_mother=[]
            for s in subset:
                cond1 = s[2]>min_overlap_area
                cond2 = area_ratio((s[3],s[4]))>min_size_similarity
                cond3 = min(s[3],s[4])>min_cell_size
                if cond1*cond2*cond3:
                    missing_mother += [s[0]]
                    init_stat[s[0]][1] += [o]
            init_stat[o][0]=missing_mother            
    rp[['mother(s)','n_mother']]=[[np.unique(init_stat[k][0]),len(np.unique(init_stat[k][0]))] for k in rp.index]
    rp[['daughter(s)','n_daughter']]=[[np.unique(init_stat[k][1]),len(np.unique(init_stat[k][1]))] for k in rp.index]
    
    
    return rp,overlap_stat

def _calculate_overlap_matrix(frame1_labeled_mask,
                              frame2_labeled_mask,
                              frame1_label,
                              frame2_label):
    f1,f2 = frame1_labeled_mask.ravel(),frame2_labeled_mask.ravel()
    f1f2 = np.array([f1,f2]).T[f1*f2 !=0]
    f1_counts = _unique2dict1D(f1)
    f2_counts = _unique2dict1D(f2)
    neighbors,overlap = np.unique(f1f2,return_counts=True,axis=0)
    f1_areas = np.array([f1_counts[i] for i in neighbors[:,0]])
    f2_areas = np.array([f2_counts[i] for i in neighbors[:,1]])
    
    f1_id = np.array(['{}_{}'.format(frame1_label,i) for i in neighbors[:,0]])
    f2_id = np.array(['{}_{}'.format(frame2_label,i) for i in neighbors[:,1]])
    overlap_matrix = np.hstack([f1_id.reshape(-1,1),
                                f2_id.reshape(-1,1),
                                overlap.reshape(-1,1),
                                f1_areas.reshape(-1,1),
                                f2_areas.reshape(-1,1),
                                (overlap/f1_areas).reshape(-1,1),
                                (overlap/f2_areas).reshape(-1,1),
                                (overlap/(f1_areas+f2_areas-overlap)).reshape(-1,1)])
    overlap_df = pd.DataFrame()
    overlap_df['frame1_id']=f1_id
    overlap_df['frame2_id']=f2_id
    overlap_df['overlap_area']=overlap
    overlap_df['frame1_area']=f1_areas
    overlap_df['frame2_area']=f2_areas
    overlap_df['frame1_overlap_frac']=overlap/f1_areas
    overlap_df['frame2_overlap_frac']=overlap/f2_areas
    overlap_df['iou']=overlap/(f1_areas+f2_areas-overlap)
    return overlap_df

def _unique2dict1D(array,nonzero=True):
    copied = array.copy()
    if nonzero:
        copied=copied[copied!=0]
    vals,counts = np.unique(copied,return_counts=True)
    return {v:c for v,c in zip(vals,counts)}

def _locate_spurious_events(regionprops,
                            n_generations=4,
                            verbose=True):
    to_be_merged = []
    to_be_split = []
    for cell in regionprops[regionprops['n_mother']>=2].index:
        back_track_record = back_track(regionprops,cell,n_generations=n_generations)
        if len(back_track_record[-1])==1:
            for entry in back_track_record:
                if len(entry)>=2:
                    if verbose:
                        print('Merge cells: {}'.format(', '.join(list(entry))))
                    to_be_merged.append(list(entry))
        else:
            if verbose:
                print('Split cell {} into {} fragments.'.format(cell,regionprops.loc[cell,'n_mother']))
            to_be_split.append([cell,regionprops.loc[cell,'n_mother']])
    return to_be_split, to_be_merged


def generate_correlation_map(x, y):
    """Correlate each n with each m.
    @abcd
    @https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays/30145770#30145770
    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def transition_matrix(frames,t,
                      moments = ['moments_hu-0','moments_hu-1','moments_hu-2',
                                  'moments_hu-3','moments_hu-4','moments_hu-5','moments_hu-6'],
                      features = ['area','major_axis_length','minor_axis_length','extent','aspect_ratio'],
                      normalize=True):
    """
    calcluate transition matrix
    channel0: Intersect(A[t],A[t+1])/A[t]
    channel1: Intersect(A[t],A[t+1])/A[t+1]
    channel2: Intersect(A[t],A[t+1])/Union(A[t],A[t+1])
    channel3: hu moments correlation matrix
    channel4: feature difference matrix
    channel5: centroid distance matrix
    channel6: nearest neighbor
    
    :params frames: momia.core.CellTracker.frames object (list)
    :params time: time
    :params features: rotation invariant features used to estimate pairwise correlation
    :return estimated transtion matrix and the two cell_id to index referenc dictionaries.
    """
    from sklearn.preprocessing import StandardScaler as ss
    f1 = frames[t]
    f2 = frames[t+1]
    n1 = len(f1.regionprops)
    n2 = len(f2.regionprops)
    f1name_dict = {'{}_{}'.format(t,x):i for i,x in enumerate(f1.regionprops.index)}
    f2name_dict = {'{}_{}'.format(t+1,x):i for i,x in enumerate(f2.regionprops.index)}

    
    # weights by overlap, channels 0-2
    transMatrix = np.zeros((n1,n2,6))
    overlap_stat = _calculate_overlap_matrix(f1.labeled_mask,f2.labeled_mask,t,t+1)
    for idx1,idx2,iou,frac1,frac2 in overlap_stat[['frame1_id','frame2_id','iou','frame1_overlap_frac','frame2_overlap_frac']].values:
        transMatrix[f1name_dict[idx1],f2name_dict[idx2],0] = iou
        transMatrix[f1name_dict[idx1],f2name_dict[idx2],1] = frac1
        transMatrix[f1name_dict[idx1],f2name_dict[idx2],2] = frac2
    
    # weights by hu moments
    m1 = f1.regionprops[moments].values
    m2 = f2.regionprops[moments].values
    
    if normalize:
        norm_v = ss().fit_transform(np.vstack([m1,m2]))
        m1 = norm_v[:n1]
        m2 = norm_v[-n2:]
    corr = generate_correlation_map(m1,m2)
    corr[corr<0]=0
    transMatrix[:,:,3] = corr
    
    v1 = f1.regionprops[features].values
    v2 = f2.regionprops[features].values
    transMatrix[:,:,4] = difference_matrix(v1,v2)
    

    cent1 = np.array([np.mean(x,axis=0) for x in f1.regionprops['$coords'].values])
    cent2 = np.array([np.mean(x,axis=0) for x in f2.regionprops['$coords'].values])
    dist = linalg.distance_matrix(cent1,cent2)
    argmin_dist = np.argmin(dist,axis=1)
    transMatrix[:,:,5] = dist_prob(dist)
    
    return transMatrix,f1name_dict,f2name_dict, overlap_stat
    

def difference_matrix(v1,v2,ref=0):
    diff = np.abs(v1[:,np.newaxis,:]-v2[np.newaxis,:,:])
    mean = (v1[:,np.newaxis,:]+v2[np.newaxis,:,:])/2
    norm_similarity = 1-np.mean(diff/mean,axis=-1)
    norm_similarity[norm_similarity<0]=0
    return norm_similarity

def dist_prob(x,D=0.1,t=1,a=0.1):
    return (1/np.sqrt(4*np.pi*t*D))*np.exp(-(a*x)**2/(4*D*t))


def silly_link(tracker_obj,t,
               iou_threshold = 0.7,
               overlap_threshold = 0.8,
                similarity_threshold = 0.95,
                diff_threshold=0.25,
                logprob_threshold = 2,
                verbose=False):

    frames = tracker_obj.frames
    trans_matrix, f1n,f2n, overlap_stat= transition_matrix(frames,t,normalize=True)
    f1 = frames[t]
    f2 = frames[t+1]
    n1 = len(f1.regionprops)
    n2 = len(f2.regionprops)
    labels1 = f1.regionprops.index
    labels2 = f2.regionprops.index
    matched_l1 = np.zeros(n1)
    matched_l2 = np.zeros(n2)

    # find stable linkages (significant overlap by IOU and shape similarity)
    linking_matrix = np.zeros((n1,n2))
    linking_matrix[(trans_matrix[:,:,2]>iou_threshold)&(trans_matrix[:,:,4]>similarity_threshold)]=1
    matched_l1[np.sum(linking_matrix,axis=1)>0]=1
    matched_l2[np.sum(linking_matrix,axis=0)>0]=1

    # fill in high-confidence links (no split/merge event)
    #"""
    f1_remnant = np.where(matched_l1==0)[0]            
    f2_remnant = np.where(matched_l2==0)[0]
    for i in f1_remnant:
        future_self_id=np.where((np.sum(np.log2(trans_matrix[i]+1),axis=1)>logprob_threshold)&(matched_l2==0))[0]
        max_prob_j = np.argmax(np.sum(np.log2(trans_matrix[i]+1),axis=1))
        if len(future_self_id)>=1: #
            area_differences = []
            for j in future_self_id:
                f1_area = f1.regionprops.loc[labels1[i],'area'].sum()
                f2_area = f2.regionprops.loc[labels2[j],'area'].sum()
                area_differences.append(np.abs(2*(f1_area-f2_area)/(f1_area+f2_area)))
            if np.min(area_differences) < diff_threshold:
                j = future_self_id[np.argmin(area_differences)]
                if j == max_prob_j:
                    linking_matrix[i,j]=1
                    matched_l1[i]=1
                    matched_l2[j]=1
    #"""
    f1_remnant = np.where(matched_l1==0)[0]            
    f2_remnant = np.where(matched_l2==0)[0]

    # find divide/merge events:
    
    for j in f2_remnant:
        former_self_id=np.where((np.sum(np.log2(trans_matrix[:,j]+1),axis=1)>logprob_threshold)&(matched_l1==0))[0]
        if len(former_self_id)>0:
            linking_matrix[former_self_id,j]=1
            f1_area = f1.regionprops.loc[labels1[former_self_id],'area'].sum()
            f2_area = f2.regionprops.loc[labels2[j],'area'].sum()
            #print(j,former_self_id,f1_area,f2_area,np.abs(2*(f1_area-f2_area)/(f1_area+f2_area)))
            if np.abs(2*(f1_area-f2_area)/(f1_area+f2_area)) < diff_threshold:
                matched_l1[former_self_id]=1
                matched_l2[j]=1
            elif f1_area<f2_area:
                matched_l1[former_self_id]=1
            elif f1_area>f2_area:
                matched_l2[j]=1
    #"""

    # assign missing daughters
    #"""
    f1_remnant = np.where(matched_l1==0)[0]   
    for i in f1_remnant:
        
        f1_area = f1.regionprops.loc[labels1[i],'area'].sum()
        f2_list = list(np.where(linking_matrix[i]>0)[0])
        top_daughters = list(np.flip(np.argsort(np.sum(np.log2(trans_matrix[i,:,np.array([0,1,2,5])]+1),axis=0)))[:4])
        top_daughters = [-1]+top_daughters
        for j in top_daughters:
            if j not in f2_list and matched_l2[j]==0 and j!=-1:
                if len(f2_list)==0:
                    f2_list.append(j)
                else:
                    j_neighbors = [f2n['{}_{}'.format(t+1,x)] for x in np.unique(np.concatenate([_get_neighbors(f2.labeled_mask,labels2[k]) for k in f2_list]))]
                    if j in j_neighbors:
                        f2_list.append(j) 
                if len(f2_list)>0:
                    f2_area = f2.regionprops.loc[labels2[np.array(f2_list)],'area'].sum()
                    if np.abs(2*(f1_area-f2_area)/(f1_area+f2_area)) < diff_threshold:
                        matched_l1[i]=1
                        matched_l2[np.array(f2_list)]=1
                        linking_matrix[i,np.array(f2_list)]=1
                        break
                    elif f2_area>f1_area:
                        f2_list.pop()

        #desperate mode:
        if matched_l1[i]==0 and len(f2_list)>0:
            for j in np.where(matched_l2==0)[0]:
                if j not in f2_list:
                    j_neighbors = [f2n['{}_{}'.format(t+1,x)] for x in np.unique(np.concatenate([_get_neighbors(f2.labeled_mask,labels2[k]) for k in f2_list]))]
                    if j in j_neighbors:
                        f2_list.append(j)    
                        f2_area = f2.regionprops.loc[labels2[np.array(f2_list)],'area'].sum()
                        if np.abs(2*(f1_area-f2_area)/(f1_area+f2_area)) < diff_threshold:
                            matched_l1[i]=1
                            matched_l2[np.array(f2_list)]=1
                            linking_matrix[i,np.array(f2_list)]=1
                            break
                        else:
                            f2_list.pop()
    #"""
    # assign missing parents
    f2_remnant = np.where(matched_l2==0)[0]   
    for j in f2_remnant:
        f2_area = f2.regionprops.loc[labels2[j],'area'].sum()
        f1_list = list(np.where(linking_matrix[:,j]>0)[0])
        top_parents = list(np.flip(np.argsort(np.sum(np.log2(trans_matrix[:,j,np.array([0,1,2,5])]+1),axis=0)))[:4])
        top_parents = [-1]+top_parents
        for i in top_parents:
            if i not in f1_list and matched_l1[i]==0 and i!=-1:
                if len(f1_list)==0:
                    f1_list.append(i)
                else:
                    i_neighbors = [f1n['{}_{}'.format(t,x)] for x in np.unique(np.concatenate([_get_neighbors(f1.labeled_mask,labels1[k]) for k in f1_list]))]
                    if i in i_neighbors:
                        f1_list.append(i)   
                if len(f1_list)>0:
                    f1_area = f1.regionprops.loc[labels1[np.array(f1_list)],'area'].sum()
                    if np.abs(2*(f1_area-f2_area)/(f1_area+f2_area)) < diff_threshold:
                        matched_l1[np.array(f1_list)]=1
                        matched_l2[j]=1
                        linking_matrix[np.array(f1_list),j]=1
                        break
                    elif f1_area>f2_area:
                        f1_list.pop()
        #desperate mode:
        if matched_l2[j]==0 and len(f1_list)>0:
            for i in np.where(matched_l1==0)[0]:
                if i not in f1_list:
                    i_neighbors = [f1n['{}_{}'.format(t,x)] for x in np.unique(np.concatenate([_get_neighbors(f1.labeled_mask,labels1[k]) for k in f1_list]))]
                    if i in i_neighbors:
                        f1_list.append(i)    
                        f1_area = f1.regionprops.loc[labels1[np.array(f1_list)],'area'].sum()
                        if np.abs(2*(f1_area-f2_area)/(f1_area+f2_area)) < diff_threshold:
                            matched_l1[np.array(f1_list)]=1
                            matched_l2[j]=1
                            linking_matrix[np.array(f1_list),j]=1
                            break
                        else:
                            f1_list.pop()


    f1_remnant = [labels1[i] for i in np.where(linking_matrix.sum(axis=1)==0)[0]] 
    f2_remnant = [labels2[j] for j in np.where(linking_matrix.sum(axis=0)==0)[0]] 
    if verbose:
        print('Unlinked items from frame {}: {}'.format(t,f1_remnant))
        print('Unlinked items from frame {}: {}'.format(t+1,f2_remnant))
    return linking_matrix,trans_matrix,overlap_stat,labels1,labels2

def _get_neighbors(labeled_mask,label):
    _x,_y = np.where(labeled_mask==label)
    x,y = dilate_by_coords(np.array([_x,_y]).T,labeled_mask.shape).T
    return [i for i in np.unique(labeled_mask[x,y]) if i not in [0,label]]