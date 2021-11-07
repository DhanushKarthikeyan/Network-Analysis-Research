/* Query to create new table available in GUI */

--select * from t_posts where forums_id = 4 and topics_id = 58582 order by posts_id;

select forums_id, count(topics_id) from t_posts group by forums_id order by forums_id;

--select * from t_posts where forums_id = 39 order by topics_id;

/* create table test as
Select forums_id, 
	count(posts_id) as TotalPosts, 
	count(distinct users_id) as UniqueUsers 
from t_posts 
	group by forums_id order by forums_id; */

/* Select count(distinct forums_id) from t_posts; */

/* Query to return total average posts and average unique users across all forums
Select avg(posts_id) as averageposts, avg(users_id) as averageusers
from t_posts
where forums_id in (
    Select forums_id  
	from t_posts 
	group by forums_id order by forums_id
	)
	and posts_id in (
    Select 
	count(posts_id) as posts_id
	
	from t_posts 
	group by forums_id order by forums_id
	)
	and users_id in (
    Select 
	count(distinct users_id) as users_id 
	from t_posts 
	group by forums_id order by forums_id
	);

	*/