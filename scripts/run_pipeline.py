# =========================================================
# run_pipeline.py
# deTACTer 전체 파이프라인 실행 스크립트
# =========================================================
# 사용법: python run_pipeline.py [--preprocess] [--extract] [--cluster] [--visualize] [--all]
# =========================================================

import argparse
import sys

# 터미널 한글 출력 인코딩 설정 (Windows)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description='deTACTer 전술 탐지 파이프라인')
    parser.add_argument('--preprocess', action='store_true', help='데이터 전처리 실행')
    parser.add_argument('--extract', action='store_true', help='시퀀스 추출 실행')
    parser.add_argument('--cluster', action='store_true', help='클러스터링 실행')
    parser.add_argument('--visualize', action='store_true', help='시각화 실행')
    parser.add_argument('--all', action='store_true', help='전체 파이프라인 실행')
    
    args = parser.parse_args()
    
    # 아무 인자도 없으면 도움말 출력
    if not any([args.preprocess, args.extract, args.cluster, args.visualize, args.all]):
        parser.print_help()
        return
    
    print("=" * 60)
    print("deTACTer 전술 탐지 파이프라인")
    print("=" * 60)
    
    if args.all or args.preprocess:
        print("\n[STEP 1] 데이터 전처리")
        print("-" * 40)
        from preprocessing import run_preprocessing
        run_preprocessing()
    
    if args.all or args.extract:
        print("\n[STEP 2] 시퀀스 추출")
        print("-" * 40)
        from sequence_extraction import run_sequence_extraction
        run_sequence_extraction()
    
    if args.all or args.cluster:
        print("\n[STEP 3] 클러스터링")
        print("-" * 40)
        from tactical_clustering import run_clustering
        run_clustering()
    
    if args.all or args.visualize:
        print("\n[STEP 4] 시각화")
        print("-" * 40)
        from visualize_tactics import run_visualization
        run_visualization()
    
    print("\n" + "=" * 60)
    print("파이프라인 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
