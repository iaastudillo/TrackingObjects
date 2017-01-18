# -*- coding: utf-8 -*-
import argparse
import os
import sys
from collections import deque
import cv2
import imutils
import numpy as np

cam = cv2.VideoCapture(0)
# print 'horizontal =', cam.get(3), 'vertical =', cam.get(4)

# Estructura para mantener una lista de ubicaciones
# nos permitirá dibujar el "contrail" de la pelota como su seguimiento.
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())
pts = deque(maxlen=args["buffer"])

centroPantalla = (215, 80)


def invertirImagen(img):
    return cv2.flip(img, 1)


def convertirHSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def transformacionesMorfologicas(mascara):
    kernel = np.ones((5, 5), "uint8")
    mascaraR = cv2.erode(mascara, kernel, iterations=2)
    mascaraR1 = cv2.dilate(mascaraR, kernel, iterations=2)
    return mascaraR1


def dibujarRectangulos(img):
    # Verde
    cv2.rectangle(img, (2, 2), (150, 150), (0, 255, 0), 3)
    # Azul
    cv2.rectangle(img, (637, 2), (487, 150), (255, 0, 0), 3)
    # Rojo
    cv2.rectangle(img, (150, 328), (2, 478), (0, 0, 255), 3)
    # Amarillo
    cv2.rectangle(img, (478, 328), (637, 478), (0, 255, 255), 3)


def buscarVerde(hsv):
    # Define los límites inferior y superior del color / Prueba y Error
    verdeMin = np.array([50, 100, 100], np.uint8)
    verdeMax = np.array([90, 255, 255], np.uint8)
    # Encuentra el rango de colores en la imange
    verde = cv2.inRange(hsv, verdeMin, verdeMax)
    return verde


def seguimientoColorVerde(img, verdeM):
    # Obtenemos los contornos
    (_, contours, hierarchy) = cv2.findContours(verdeM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    if len(contours) > 0:
        # Busca el contorno mas grande
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            # Dibuja el circulo y el punto central
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 0), 2)
            cv2.circle(img, center, 5, (255, 255, 255), -1)
            cv2.putText(img, "VERDE", (int(x - radius), int(y - (radius + 10))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            # Verifica que el objeto se encuentre en el cuadro correcto
            if x < 150 and y < 150:
                cv2.putText(img, "Bien :)", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x > 487 and y < 150:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x < 150 and y > 328:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x > 487 and y > 328:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

            dibujarSeguimiento(center, img)


def buscarAzul(hsv):
    azulMin = np.array([120, 95, 95], np.uint8)
    azulMax = np.array([130, 255, 255], np.uint8)
    azul = cv2.inRange(hsv, azulMin, azulMax)
    return azul


def seguimientoColorAzul(img, azulM):
    (_, contours, hierarchy) = cv2.findContours(azulM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 0), 2)
            cv2.circle(img, center, 5, (255, 255, 255), -1)
            cv2.putText(img, "AZUL", (int(x - radius), int(y - (radius + 10))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
            if x > 487 and y < 150:
                cv2.putText(img, "Bien :)", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x < 150 and y < 150:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x < 150 and y > 328:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x > 487 and y > 328:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

                # dibujarSeguimiento(center, img)


def buscarRojo(hsv):
    rojoMin = np.array([150, 135, 135], np.uint8)
    rojoMax = np.array([180, 255, 255], np.uint8)
    rojo = cv2.inRange(hsv, rojoMin, rojoMax)
    return rojo


def seguimientoColorRojo(img, rojoM):
    (_, contours, hierarchy) = cv2.findContours(rojoM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 0), 2)
            cv2.circle(img, center, 5, (255, 255, 255), -1)
            cv2.putText(img, "ROJO", (int(x - radius), int(y - (radius + 5))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            if x < 150 and y > 328:
                cv2.putText(img, "Bien :)", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x < 150 and y < 150:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x > 487 and y < 150:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x > 487 and y > 328:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

        dibujarSeguimiento(center, img)


def buscarAmarillo(hsv):
    amarilloMin = np.array([20, 80, 80], np.uint8)
    amarilloMax = np.array([40, 255, 255], np.uint8)
    amarillo = cv2.inRange(hsv, amarilloMin, amarilloMax)
    return amarillo


def seguimientoColorAmarillo(img, amarilloM):
    (_, contours, hierarchy) = cv2.findContours(amarilloM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 0), 2)
            cv2.circle(img, center, 5, (255, 255, 255), -1)
            cv2.putText(img, "AMARILLO", (int(x - radius), int(y - (radius + 10))), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            if x > 487 and y > 328:
                cv2.putText(img, "Bien :)", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x < 150 and y < 150:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x > 487 and y < 150:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            if x < 150 and y > 328:
                cv2.putText(img, "Mal :(", centroPantalla, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

                # dibujarSeguimiento(center, img)


def dibujarSeguimiento(center, img):
    pts.appendleft(center)
    for i in xrange(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(img, pts[i - 1], pts[i], (255, 255, 255), thickness)


while (1):

    _, img1 = cam.read()

    img = invertirImagen(img1)

    hsv = convertirHSV(img)

    verdeE = buscarVerde(hsv)
    verdeM = transformacionesMorfologicas(verdeE)
    seguimientoColorVerde(img, verdeM)

    azulB = buscarAzul(hsv)
    azulM = transformacionesMorfologicas(azulB)
    seguimientoColorAzul(img, azulM)

    rojoB = buscarRojo(hsv)
    rojoM = transformacionesMorfologicas(rojoB)
    seguimientoColorRojo(img, rojoM)

    amarilloB = buscarAmarillo(hsv)
    amarilloM = transformacionesMorfologicas(amarilloB)
    seguimientoColorAmarillo(img, amarilloM)

    dibujarRectangulos(img)

    cv2.imshow("OBJECT TRACKING", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cam.release()
        cv2.destroyAllWindows()
        break
