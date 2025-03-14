from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/sim6/zlg/data/sets/nuscenes/v1.0-trainval_meta', verbose=True)
nusc.list_scenes()
