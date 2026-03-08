#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging("Fantasy");
  FLAGS_alsologtostderr = true;
  FLAGS_log_dir = "./log";
  mkdir(FLAGS_log_dir.c_str(), 0755);

  LOG(INFO) << "Start test...\n";
  return RUN_ALL_TESTS();
}