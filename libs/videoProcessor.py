import cv2
import concurrent.futures
from libs.imageExtractor import proccessFrames



def process_video(path):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    frames = []
    counters = []
    batchPositions = []
    # ocrModel = loadModel()
    try:
        video = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not video.isOpened():
            raise ValueError(f"No se pudo abrir el video: {path}")
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        batch_size = int(fps)
        frame_cut_index=2
        print(f"precision de 1/{frame_cut_index}" )
        print(f"Total de frames: {total_frames}, FPS: {fps}")
        frame_counter = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if(frame_counter%frame_cut_index==0):
                frames.append(frame)
                counters.append(frame_counter)
                if len(frames) == batch_size:
                    proccessFrames(frames,batch_size,counters,fps)
                    # pool.submit(extractText,framePositions,'test.srt',ocrModel)
                    # batchPositions.extend(framePositions)
                    # pool.submit(proccessFrames,[frames,batch_size,counters])
                    frames = []
                    counters = []
            frame_counter+=1
        video.release()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # pool.shutdown(wait=True)
        cv2.destroyAllWindows()
        