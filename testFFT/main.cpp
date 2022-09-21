#include <iostream> 
#include <cmath>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctype.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    //����
    Mat image = cv::imread("hpk.bmp", cv::IMREAD_GRAYSCALE);

    //�t�[���G�ϊ��pMat
    Mat furimg;

    //�����݂̂�image�Ƌ�����0�ŏ���������Mat��RealImaginary�z��ɓ����
    Mat RealIamginary[] = { Mat_<float>(image), Mat::zeros(image.size(), CV_32F) };

    //�z�������
    merge(RealIamginary, 2, furimg);

    //�t�[���G�ϊ�
    dft(furimg, furimg);

    //�\���p
    Mat divdisplay[2];

    //�t�[���G��������Ƌ����ɕ�����
    split(furimg, divdisplay);

    //�\���p�ɂ��ׂĎ�����
    Mat display;
    magnitude(divdisplay[0], divdisplay[1], display);

    //�ΐ��ɕϊ�����i���̂��ߊe�s�N�Z���ɂP�����Z�j
    display += Scalar::all(1);
    log(display, display);

    //�������牺��ǉ�
    //___________________________________________

    const int halfW = display.cols / 2;
    const int halfH = display.rows / 2;

    Mat tmp;

    Mat q0(display,
        Rect(0, 0, halfW, halfH));
    Mat q1(display,
        Rect(halfW, 0, halfW, halfH));
    Mat q2(display,
        Rect(0, halfH, halfW, halfH));
    Mat q3(display,
        Rect(halfW, halfH, halfW, halfH));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    //____________________________________________
    //����������ǉ�

    //�\���p�ɐ��K��
    Mat outdisplay;
    normalize(display, outdisplay, 0, 1, NORM_MINMAX);

    namedWindow("aftdft");
    imshow("aftdft", outdisplay);
    imwrite("hpk11.bmp", outdisplay);

    waitKey(-1);

    return 0;

}



/*
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;
static void help(void)
{
    cout << endl
        << "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        << "The dft of an image is taken and it's power spectrum is displayed." << endl
        << "Usage:" << endl
        << "./discrete_fourier_transform [image_name -- default ../data/lena.jpg]" << endl;
}
int main(int argc, char** argv)
{
    help();
    const char* filename = argc >= 2 ? argv[1] : "HPK.bmp";
    Mat I = imread(filename, IMREAD_GRAYSCALE);
    if (I.empty()) {
        cout << "Error opening image" << endl;
        return -1;
    }
    Mat padded;
    // ���͉摜��DFT�ɍœK�ȑ傫���ɍL����B�摜�T�C�Y��2,3,5�̔{���̂Ƃ��ɍ����ɂȂ�
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols);
    // ���͉摜�𒆉��ɒu���A���͂�0�Ŗ��߂�
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    //dtf�̌��ʂ͕��f���i�P�v�f�ɂ��Q�l������j�ł���A�܂��l�̕\���͈͂��L���B
    //���̂���float�ƕ��f����ێ�������̂�2�����
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    // 2���̉摜���A2�`���l����������1���ɂ���
    merge(planes, 2, complexI);
    // complexI��dft��K�p���AcomplexI�Ɍ��ʂ�߂�
    dft(complexI, complexI);

    // ��Βl���v�Z���Alog�X�P�[���ɂ���
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1);
    // ���ʂ̒l�͑傫��������̂Ə�����������̂��������Ă���̂ŁAlog��K�p���ė}������                  
    log(magI, magI);
    // �s�E�񂪊�̏ꍇ�p�B�N���b�v����
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // �摜�̒����Ɍ��_������悤�ɁA�ی������ւ���(���̂܂܂��Ɖ摜�̒��S�������S�p�̕����������Ă���炵���H)
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // ����i���ی��j
    Mat q1(magI, Rect(cx, 0, cx, cy));  // �E��i���ی��j
    Mat q2(magI, Rect(0, cy, cx, cy));  // �����i��O�ی��j
    Mat q3(magI, Rect(cx, cy, cx, cy)); // �E���i��l�ی��j
    Mat tmp;
    // ����ւ�(����ƉE��)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    // �E��ƍ���
    q2.copyTo(q1);
    tmp.copyTo(q2);
    // ���邱�Ƃ��ł���l(float[0,1])�ɕϊ�
    imwrite("HPK1.bmp", magI);
    normalize(magI, magI, 0, 1, NORM_MINMAX);



    //�\��     
    imshow("Input Image", I);
    imshow("spectrum magnitude", magI);
    waitKey();
    return 0;
}

*/