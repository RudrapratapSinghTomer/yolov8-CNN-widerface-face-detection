from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import matplotlib.pyplot as plt

class Evulation:
    def __init__(self):
        pass

    def lou_(self, box1, box2):
        x0A, y0A, x1A, y1A = box1
        x0B, y0B, x1B, y1B = box2

        x0I = max((x0A,x0B))
        y0I = max((y0A,y0B))
        x1I = min((x1A,x1B))
        y1I = min((y1A,y1B))

        intersection_area = max(0, (x1I-x0I)) * max(0, (y1I-y0I))
        area_box1 = (x1A - y1A) * (x0A - y0A)
        area_box2 = (x1B - y1B) * (x0B - y0B)
        union_area = area_box1 + area_box2 - intersection_area

        iou = intersection_area/union_area

        return iou
    
    def match_predication(self, predict_frame, actual_frame, threshold=0.5):
        matched_gt = set()
        TP = 0
        FP = 0
        TOTAL_BACKGROUND = 10000

        for pred_box in predict_frame:
            best_iou = 0
            best_iou_idx = -1

            for idx, gt_box in enumerate(actual_frame):
                if idx in matched_gt:
                    continue
                
                iou_abc = self.lou_(pred_box, gt_box)

                if iou_abc > best_iou:
                    best_iou = iou_abc
                    best_iou_idx = idx


            if iou_abc >= threshold:
                TP += 1
                matched_gt.add(best_iou_idx)
            else:
                    FP =+ 1

        FN = len(actual_frame) - len(predict_frame)
        TN = TOTAL_BACKGROUND - FP

        return TP, TN, FP, FN
    
    # def precision(TP, FP):
    #     return TP/(TP+FP)
    
    def precision(self, TP, FP):
        return TP / (TP + FP) if (TP + FP) > 0 else 0

    
    # def recall(TP, FN):
        # return TP/(TP+FN)
    
    def recall(self, TP, FN):
        return TP / (TP + FN) if (TP + FN) > 0 else 0

    
    def compute_PRcurve(self, predict_frame, actual_frame, predict_score):
        precisions = []
        recalls = []
        
        for i in range(predict_frame):
            for j in range(actual_frame):
                TP, FP, FN = self.match_predication(predict_frame[i], actual_frame[j], threshold=0.5)

                precision = self.precision(TP, FP)
                recall = self.recall(TP, FN)

                precisions.append(precision)
                recalls.append(recall)

        return recalls, precisions
    
    # def compute_ROCcurve(TP, TN, FP, FN):
    #     TPR = TP / (TP + FN)
    #     FPR = FP / (FP + TN)
    #     return TPR, FPR
    
    def compute_ROCcurve(self, TP, TN, FP, FN):
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        return TPR, FPR
    
    def plotPRcurve(self, recall, precision):
        plt.figure()
        plt.plot(recall, precision, marker='o')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.show()
        

    # def plotROCcurve(FPR, TPR):
    #     plt.figure()
    #     plt.plot(FPR, TPR, marker='o')
    #     plt.xlabel('FPR')
    #     plt.ylabel('TPR')
    #     plt.title("ROC Curve")
    #     plt.grid(True)
    #     plt.show()

    def plotROCcurve(self, TPR, FPR):

    # handle single-point case
        if not isinstance(TPR, (list, tuple)):
            TPR = [TPR]
            FPR = [FPR]

        plt.figure()
        plt.plot(FPR, TPR, marker='o')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.show()