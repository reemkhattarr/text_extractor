def main():
    try:
        with open("evaluation_results.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        print(f"Read {len(lines)} lines.")
        for line in lines:
            print(line.rstrip())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
