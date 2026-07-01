#pragma once
#include <rust-lib/lib.h>

bool isWvr8(const rust::Vec<uint8_t>& data);
void populateFromWvr8(const rust::Vec<uint8_t>& data, arma_file_formats::cxx::OprwCxx& wrp);
