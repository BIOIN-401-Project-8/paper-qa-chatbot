import paperscraper

def main():
    keyword_search = 'bispecific antibody manufacture'
    papers = paperscraper.search_papers(keyword_search, limit=2)
    for path, data in papers.items():
        print(path)


if __name__ == '__main__':
    main()
