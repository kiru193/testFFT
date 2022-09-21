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
    //入力
    Mat image = cv::imread("hpk.bmp", cv::IMREAD_GRAYSCALE);

    //フーリエ変換用Mat
    Mat furimg;

    //実部のみのimageと虚部を0で初期化したMatをRealImaginary配列に入れる
    Mat RealIamginary[] = { Mat_<float>(image), Mat::zeros(image.size(), CV_32F) };

    //配列を合成
    merge(RealIamginary, 2, furimg);

    //フーリエ変換
    dft(furimg, furimg);

    //表示用
    Mat divdisplay[2];

    //フーリエ後を実部と虚部に分ける
    split(furimg, divdisplay);

    //表示用にすべて実数に
    Mat display;
    magnitude(divdisplay[0], divdisplay[1], display);

    //対数に変換する（そのため各ピクセルに１を加算）
    display += Scalar::all(1);
    log(display, display);

    //ここから下を追加
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
    //ここから上を追加

    //表示用に正規化
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
    // 入力画像をDFTに最適な大きさに広げる。画像サイズが2,3,5の倍数のときに高速になる
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols);
    // 入力画像を中央に置き、周囲は0で埋める
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    //dtfの結果は複素数（１要素につき２つ値がある）であり、また値の表現範囲が広い。
    //そのためfloatと複素数を保持するもので2枚作る
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    // 2枚の画像を、2チャネルを持った1枚にする
    merge(planes, 2, complexI);
    // complexIにdftを適用し、complexIに結果を戻す
    dft(complexI, complexI);

    // 絶対値を計算し、logスケールにする
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1);
    // 結果の値は大きすぎるものと小さすぎるものが混じっているので、logを適用して抑制する                  
    log(magI, magI);
    // 行・列が奇数の場合用。クロップする
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // 画像の中央に原点が来るように、象限を入れ替える(元のままだと画像の中心部分が４つ角の方向を向いているらしい？)
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // 左上（第二象限）
    Mat q1(magI, Rect(cx, 0, cx, cy));  // 右上（第一象限）
    Mat q2(magI, Rect(0, cy, cx, cy));  // 左下（第三象限）
    Mat q3(magI, Rect(cx, cy, cx, cy)); // 右下（大四象限）
    Mat tmp;
    // 入れ替え(左上と右下)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    // 右上と左下
    q2.copyTo(q1);
    tmp.copyTo(q2);
    // 見ることができる値(float[0,1])に変換
    imwrite("HPK1.bmp", magI);
    normalize(magI, magI, 0, 1, NORM_MINMAX);



    //表示     
    imshow("Input Image", I);
    imshow("spectrum magnitude", magI);
    waitKey();
    return 0;
}

*/