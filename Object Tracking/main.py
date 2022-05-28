# load libraries
import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# main block
if __name__ == '__main__' :
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    tracker_type = tracker_types[7]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)

    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == 'CSRT':
            tracker = cv2.TrackerCSRT_create()

    # read video
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Could not open camera")
        sys.exit()

    ok, frame = video.read()
    if not ok :
        print("Cnnot read the video file")
        sys.exit()

    # create object box
    bbox = (287, 23, 86, 320)
    bbox = cv2.selectROI(frame, False)

    ok = tracker.init(frame, bbox)

    while True:
        # read new frame
        ok, frame = video.read()
        if not ok:
            break

        # start timer
        timer = cv2.getTickCount()

        # update tracker
        ok, bbox = tracker.update(frame)

        # calculate frames per seconds
        fps = cv2.getThickFrequency()/(cv2.getThickCount - timer)

        # draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

        cv2.putText(frame, tracker_type + "tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        cv2.putText(frame, 'FPS: ' + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    video.release()
    cv2.destroyAllWindows()
