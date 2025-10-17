#pragma once
#ifndef SJTU_BIGINTEGER
#define SJTU_BIGINTEGER
// Combined header + implementation for OJ submission
// Only permitted headers
#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

namespace sjtu {
class int2048 {
private:
  static const int BASE = 10000;       // 10^4
  static const int BASE_DIGS = 4;      // digits per limb
  std::vector<int> limbs;              // little-endian limbs
  bool negative = false;               // sign flag, false for >= 0

  void normalize();
  static int compareAbs(const int2048 &a, const int2048 &b);
  static void addAbsTo(int2048 &a, const int2048 &b);
  static void subAbsTo(int2048 &a, const int2048 &b);

  static int2048 multiplySchoolbook(const int2048 &a, const int2048 &b);
  static int2048 multiplyFFT(const int2048 &a, const int2048 &b);

  static void divmodAbs(const int2048 &a, const int2048 &b, int2048 &q, int2048 &r);

public:
  int2048();
  int2048(long long);
  int2048(const std::string &);
  int2048(const int2048 &);

  void read(const std::string &);
  void print();

  int2048 &add(const int2048 &);
  friend int2048 add(int2048, const int2048 &);

  int2048 &minus(const int2048 &);
  friend int2048 minus(int2048, const int2048 &);

  int2048 operator+() const;
  int2048 operator-() const;

  int2048 &operator=(const int2048 &);

  int2048 &operator+=(const int2048 &);
  friend int2048 operator+(int2048, const int2048 &);

  int2048 &operator-=(const int2048 &);
  friend int2048 operator-(int2048, const int2048 &);

  int2048 &operator*=(const int2048 &);
  friend int2048 operator*(int2048, const int2048 &);

  int2048 &operator/=(const int2048 &);
  friend int2048 operator/(int2048, const int2048 &);

  int2048 &operator%=(const int2048 &);
  friend int2048 operator%(int2048, const int2048 &);

  friend std::istream &operator>>(std::istream &, int2048 &);
  friend std::ostream &operator<<(std::ostream &, const int2048 &);

  friend bool operator==(const int2048 &, const int2048 &);
  friend bool operator!=(const int2048 &, const int2048 &);
  friend bool operator<(const int2048 &, const int2048 &);
  friend bool operator>(const int2048 &, const int2048 &);
  friend bool operator<=(const int2048 &, const int2048 &);
  friend bool operator>=(const int2048 &, const int2048 &);
};

// ===== implementation =====
void int2048::normalize() {
  while (!limbs.empty() && limbs.back() == 0) limbs.pop_back();
  if (limbs.empty()) negative = false;
}

int int2048::compareAbs(const int2048 &a, const int2048 &b) {
  if (a.limbs.size() != b.limbs.size())
    return a.limbs.size() < b.limbs.size() ? -1 : 1;
  for (int i = (int)a.limbs.size() - 1; i >= 0; --i) {
    if (a.limbs[i] != b.limbs[i]) return a.limbs[i] < b.limbs[i] ? -1 : 1;
  }
  return 0;
}

void int2048::addAbsTo(int2048 &a, const int2048 &b) {
  int carry = 0;
  size_t n = a.limbs.size();
  size_t m = b.limbs.size();
  size_t len = n > m ? n : m;
  if (a.limbs.size() < len) a.limbs.resize(len, 0);
  for (size_t i = 0; i < len; ++i) {
    long long sum = (long long)a.limbs[i] + (i < m ? b.limbs[i] : 0) + carry;
    a.limbs[i] = (int)(sum % BASE);
    carry = (int)(sum / BASE);
  }
  if (carry) a.limbs.push_back(carry);
}

void int2048::subAbsTo(int2048 &a, const int2048 &b) {
  int carry = 0;
  size_t m = b.limbs.size();
  for (size_t i = 0; i < a.limbs.size(); ++i) {
    long long diff = (long long)a.limbs[i] - (i < m ? b.limbs[i] : 0) - carry;
    if (diff < 0) {
      diff += BASE;
      carry = 1;
    } else {
      carry = 0;
    }
    a.limbs[i] = (int)diff;
  }
  a.normalize();
}

int2048 int2048::multiplySchoolbook(const int2048 &a, const int2048 &b) {
  int2048 res;
  if (a.limbs.empty() || b.limbs.empty()) return res;
  res.limbs.assign(a.limbs.size() + b.limbs.size(), 0);
  long long base = BASE;
  for (size_t i = 0; i < a.limbs.size(); ++i) {
    long long carry = 0;
    for (size_t j = 0; j < b.limbs.size() || carry; ++j) {
      long long cur = res.limbs[i + j] + carry;
      if (j < b.limbs.size()) cur += 1LL * a.limbs[i] * b.limbs[j];
      res.limbs[i + j] = (int)(cur % base);
      carry = cur / base;
    }
  }
  res.normalize();
  return res;
}

static void fft(std::vector<std::complex<double>> &a, bool invert) {
  size_t n = a.size();
  static std::vector<size_t> rev;
  static size_t last_n = 0;
  if (n != last_n) {
    rev.assign(n, 0);
    for (size_t i = 1, j = 0; i < n; ++i) {
      size_t bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      rev[i] = j;
    }
    last_n = n;
  }
  for (size_t i = 0; i < n; ++i) if (i < rev[i]) { auto tmp=a[i]; a[i]=a[rev[i]]; a[rev[i]]=tmp; }
  for (size_t len = 2; len <= n; len <<= 1) {
    const double PI = 3.14159265358979323846264338327950288;
    double ang = 2 * PI / (double)len * (invert ? -1.0 : 1.0);
    std::complex<double> wlen = std::polar(1.0, ang);
    for (size_t i = 0; i < n; i += len) {
      std::complex<double> w(1);
      for (size_t j = 0; j < len / 2; ++j) {
        std::complex<double> u = a[i + j];
        std::complex<double> v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
  if (invert) {
    for (size_t i = 0; i < n; ++i) a[i] /= (double)n;
  }
}

int2048 int2048::multiplyFFT(const int2048 &a, const int2048 &b) {
  const size_t THRESH = 64;
  if (a.limbs.size() < THRESH || b.limbs.size() < THRESH)
    return multiplySchoolbook(a, b);

  std::vector<std::complex<double>> fa(a.limbs.begin(), a.limbs.end());
  std::vector<std::complex<double>> fb(b.limbs.begin(), b.limbs.end());
  size_t n = 1;
  while (n < fa.size() + fb.size()) n <<= 1;
  fa.resize(n); fb.resize(n);
  fft(fa, false); fft(fb, false);
  for (size_t i = 0; i < n; ++i) fa[i] *= fb[i];
  fft(fa, true);

  int2048 res;
  res.limbs.assign(n, 0);
  long long carry = 0;
  for (size_t i = 0; i < n; ++i) {
    double val = fa[i].real();
    long long rounded = (long long)(val + (val >= 0 ? 0.5 : -0.5));
    long long t = rounded + carry;
    res.limbs[i] = (int)(t % BASE);
    carry = t / BASE;
  }
  while (carry) {
    res.limbs.push_back((int)(carry % BASE));
    carry /= BASE;
  }
  res.normalize();
  return res;
}

void int2048::divmodAbs(const int2048 &a, const int2048 &b, int2048 &q, int2048 &r) {
  q.limbs.clear(); q.negative = false;
  r.limbs.clear(); r.negative = false;
  if (b.limbs.empty()) return;
  q.limbs.assign(a.limbs.size(), 0);
  int2048 cur;
  for (int i = (int)a.limbs.size() - 1; i >= 0; --i) {
    if (!cur.limbs.empty() || a.limbs[i] != 0) {
      cur.limbs.insert(cur.limbs.begin(), 0);
    }
    if (!cur.limbs.empty()) cur.limbs[0] = a.limbs[i];
    else cur.limbs.push_back(a.limbs[i]);
    cur.normalize();
    int low = 0, high = BASE - 1, best = 0;
    while (low <= high) {
      int mid = (low + high) >> 1;
      int2048 prod;
      if (mid == 0) prod = int2048(0);
      else {
        prod.limbs.assign(b.limbs.size() + 1, 0);
        long long carry = 0;
        for (size_t j = 0; j < b.limbs.size() || carry; ++j) {
          long long curv = (j < b.limbs.size() ? 1LL * b.limbs[j] * mid : 0) + carry;
          if (j >= prod.limbs.size()) prod.limbs.push_back(0);
          prod.limbs[j] = (int)(curv % BASE);
          carry = curv / BASE;
        }
        prod.normalize();
      }
      int cmp = compareAbs(prod, cur);
      if (cmp <= 0) {
        best = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    q.limbs[i] = best;
    if (best != 0) {
      int2048 prod;
      prod.limbs.assign(b.limbs.size() + 1, 0);
      long long carry = 0;
      for (size_t j = 0; j < b.limbs.size() || carry; ++j) {
        long long curv = (j < b.limbs.size() ? 1LL * b.limbs[j] * best : 0) + carry;
        if (j >= prod.limbs.size()) prod.limbs.push_back(0);
        prod.limbs[j] = (int)(curv % BASE);
        carry = curv / BASE;
      }
      prod.normalize();
      subAbsTo(cur, prod);
    }
  }
  q.normalize();
  r = cur; r.negative = false; r.normalize();
}

int2048::int2048() { negative = false; }

int2048::int2048(long long v) {
  negative = false;
  if (v < 0) { negative = true; v = -v; }
  while (v) {
    limbs.push_back((int)(v % BASE));
    v /= BASE;
  }
  normalize();
}

int2048::int2048(const std::string &s) { read(s); }

int2048::int2048(const int2048 &o) {
  limbs = o.limbs; negative = o.negative;
}

void int2048::read(const std::string &s) {
  limbs.clear(); negative = false;
  size_t i = 0; while (i < s.size() && isspace((unsigned char)s[i])) ++i;
  if (i < s.size() && (s[i] == '+' || s[i] == '-')) { negative = (s[i] == '-'); ++i; }
  while (i < s.size() && s[i] == '0') ++i;
  std::vector<int> digits;
  for (; i < s.size(); ++i) {
    if (isdigit((unsigned char)s[i])) digits.push_back(s[i] - '0');
  }
  if (digits.empty()) { negative = false; return; }
  int cur = 0, cnt = 0;
  for (int d : digits) {
    cur = cur * 10 + d;
    ++cnt;
    if (cnt == BASE_DIGS) {
      limbs.insert(limbs.begin(), cur);
      cur = 0; cnt = 0;
    }
  }
  if (cnt != 0) limbs.insert(limbs.begin(), cur);
  std::reverse(limbs.begin(), limbs.end());
  normalize();
}

void int2048::print() {
  if (limbs.empty()) { std::cout << 0; return; }
  if (negative) std::cout << '-';
  int n = (int)limbs.size();
  std::cout << limbs.back();
  char buf[16];
  for (int i = n - 2; i >= 0; --i) {
    std::snprintf(buf, sizeof(buf), "%0*d", BASE_DIGS, limbs[i]);
    std::cout << buf;
  }
}

int2048 &int2048::add(const int2048 &other) {
  if (other.limbs.empty()) return *this;
  if (limbs.empty()) { *this = other; return *this; }
  if (negative == other.negative) {
    addAbsTo(*this, other);
  } else {
    int cmp = compareAbs(*this, other);
    if (cmp >= 0) {
      subAbsTo(*this, other);
    } else {
      int2048 tmp = other;
      subAbsTo(tmp, *this);
      *this = tmp;
    }
  }
  return *this;
}

int2048 add(int2048 a, const int2048 &b) { return a.add(b); }

int2048 &int2048::minus(const int2048 &other) {
  if (other.limbs.empty()) return *this;
  int2048 negOther = other; negOther.negative = !negOther.negative;
  return this->add(negOther);
}

int2048 minus(int2048 a, const int2048 &b) { return a.minus(b); }

int2048 int2048::operator+() const { return *this; }
int2048 int2048::operator-() const { int2048 t(*this); if (!t.limbs.empty()) t.negative = !t.negative; return t; }

int2048 &int2048::operator=(const int2048 &o) { if (this != &o) { limbs = o.limbs; negative = o.negative; } return *this; }

int2048 &int2048::operator+=(const int2048 &o) { return add(o); }
int2048 operator+(int2048 a, const int2048 &b) { return a += b; }

int2048 &int2048::operator-=(const int2048 &o) { return minus(o); }
int2048 operator-(int2048 a, const int2048 &b) { return a -= b; }

int2048 &int2048::operator*=(const int2048 &o) {
  if (limbs.empty() || o.limbs.empty()) { limbs.clear(); negative = false; return *this; }
  int sign = (negative ^ o.negative) ? -1 : 1;
  int2048 res = multiplyFFT(*this, o);
  *this = res;
  negative = (sign < 0) && !limbs.empty();
  return *this;
}
int2048 operator*(int2048 a, const int2048 &b) { return a *= b; }

int2048 &int2048::operator/=(const int2048 &o) {
  int sign_neg = (negative ^ o.negative);
  int2048 A = *this; A.negative = false;
  int2048 B = o; B.negative = false;
  int2048 q, r;
  divmodAbs(A, B, q, r);
  if (sign_neg && !r.limbs.empty()) {
    int carry = 1;
    for (size_t i = 0; i < q.limbs.size() && carry; ++i) {
      int v = q.limbs[i] + carry;
      if (v >= BASE) { q.limbs[i] = v - BASE; carry = 1; }
      else { q.limbs[i] = v; carry = 0; }
    }
    if (carry) q.limbs.push_back(carry);
  }
  q.negative = sign_neg && !q.limbs.empty();
  q.normalize();
  *this = q;
  return *this;
}
int2048 operator/(int2048 a, const int2048 &b) { return a /= b; }

int2048 &int2048::operator%=(const int2048 &o) {
  int2048 q = *this / o;
  int2048 prod = q * o;
  *this -= prod;
  return *this;
}
int2048 operator%(int2048 a, const int2048 &b) { return a %= b; }

std::istream &operator>>(std::istream &in, int2048 &x) {
  std::string s; in >> s; x.read(s); return in;
}
std::ostream &operator<<(std::ostream &out, const int2048 &x) {
  if (x.limbs.empty()) { out << 0; return out; }
  if (x.negative) out << '-';
  out << x.limbs.back();
  char buf[16];
  for (int i = (int)x.limbs.size() - 2; i >= 0; --i) {
    std::snprintf(buf, sizeof(buf), "%0*d", int2048::BASE_DIGS, x.limbs[i]);
    out << buf;
  }
  return out;
}

bool operator==(const int2048 &a, const int2048 &b) {
  return a.negative == b.negative && a.limbs == b.limbs;
}
bool operator!=(const int2048 &a, const int2048 &b) { return !(a == b); }
bool operator<(const int2048 &a, const int2048 &b) {
  if (a.negative != b.negative) return a.negative;
  int cmp = int2048::compareAbs(a, b);
  return a.negative ? (cmp > 0) : (cmp < 0);
}
bool operator>(const int2048 &a, const int2048 &b) { return b < a; }
bool operator<=(const int2048 &a, const int2048 &b) { return !(b < a); }
bool operator>=(const int2048 &a, const int2048 &b) { return !(a < b); }

} // namespace sjtu

#endif
