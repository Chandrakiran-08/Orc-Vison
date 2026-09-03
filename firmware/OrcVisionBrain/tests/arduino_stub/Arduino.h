// Minimal Arduino API stub so an .ino can be compiled and run on the host.
#pragma once
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#define F(x) x
#define HIGH 1
#define LOW 0
#define OUTPUT 1
#define LED_BUILTIN 13
static inline unsigned long millis() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (unsigned long)(ts.tv_sec * 1000UL + ts.tv_nsec / 1000000UL);
}
static inline void delay(unsigned long) {}
static inline void pinMode(int, int) {}
static inline void digitalWrite(int, int) {}
struct SerialStub {
  operator bool() const { return true; }
  void begin(long) {}
  void print(const char* s) { printf("%s", s); }
  void print(unsigned v) { printf("%u", v); }
  void print(int v) { printf("%d", v); }
  void println() { printf("\n"); }
  void println(const char* s) { printf("%s\n", s); }
  void println(unsigned v) { printf("%u\n", v); }
  void println(int v) { printf("%d\n", v); }
};
static SerialStub Serial;
