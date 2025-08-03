.DEFAULT_GOAL: all
.PHONY: clean

BUILD_DIR   := build
SRC_DIR	    := src
INCLUDES    := -I./src
LIBS		:= -lm
CC          := gcc
CFLAGS      := -Wall -Wextra -O2 -MMD -MP $(LIBS) $(INCLUDES)
BIN			:= main

SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(SRC:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/$(BIN)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: clean

