
// Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
//
// This source code is licensed under the Apache-2.0 license found in the
// LICENSE file in the root directory of this source tree.

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/fcntl.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int fd = open(argv[1], O_RDONLY | O_NONBLOCK);
    if (fd < 0) {
        exit(1);
    }
    int cnt = 0;
    double sum = 0;
    while (1) {
        char buf[31];
        lseek(fd, 0, 0);
        int n = read(fd, buf, 32);
        if (n > 0) {
            buf[n] = 0;
            char *o = NULL;
            sum += strtod(buf, &o);
            cnt += 1;
        }
        if (cnt >= 1000) {
            fprintf(stderr, "%.1f ", sum / cnt);
            cnt = 0;
            sum = 0;
        }
    }
    return 0;
}